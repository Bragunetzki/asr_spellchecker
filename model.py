import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import BertConfig, BertModel

from dataloader import CustomTestDataloader
from dataset import CustomTestDataset
from input_building import InputBuilder


class AsrSpellchecker(nn.Module):
    def __init__(self, cfg):
        super(AsrSpellchecker, self).__init__()
        self.cfg = cfg
        self.num_labels = 743
        self.bert_hidden_size = 768

        # Initialize layers
        self.linear = nn.Linear(self.bert_hidden_size * 2, self.num_labels)
        self.logits_dropout = nn.Dropout(0.1)
        self.loss_function = nn.CrossEntropyLoss()

        # Initialize BERT
        bert_config = BertConfig.from_json_file(cfg['bert_cfg_path'])
        self.bert_model = BertModel.from_pretrained(cfg['bert_model_name'], config=bert_config,
                                                    ignore_mismatched_sizes=True)

        # Move to device
        self.to(cfg['device'])
        self.example_builder = InputBuilder(cfg)

        params = self.bert_model.parameters()
        self.optimizer = optim.Adam(params, lr=0.00003)

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.batch_accuracy = []
        self.validation_acc_vals = [0] * 6

    @staticmethod
    def tensor2list(tensor):
        """ Converts tensor to a list """
        return tensor.detach().cpu().tolist()

    def forward(
            self,
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos
    ):
        # BERT embeddings for characters; shape = [batch_size, seq_len, hidden_size]
        src_hiddens = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)[0]
        # BERT embeddings for subwords; shape = [batch_size, subword_seq_len, hidden_size]
        src_hiddens_for_subwords = self.bert_model(
            input_ids=input_ids_for_subwords,
            token_type_ids=segment_ids_for_subwords,
            attention_mask=input_mask_for_subwords,
        )[0]

        # Concatenate subword embeddings to character embeddings
        index = character_pos_to_subword_pos.unsqueeze(-1).expand((-1, -1, src_hiddens_for_subwords.shape[2]))
        src_hiddens_2 = torch.gather(src_hiddens_for_subwords, 1, index)
        src_hiddens = torch.cat((src_hiddens, src_hiddens_2), 2)

        # Compute logits
        src_hiddens = self.logits_dropout(src_hiddens)
        logits = self.linear(src_hiddens)
        return logits

    def make_infer_data(self, input_name):
        input_examples, all_hypotheses = self.example_builder.build_input_from_file(input_name, True)
        dataset = CustomTestDataset(self.cfg['tokenizer'].pad_token_id, input_examples, all_hypotheses)
        dataloader = CustomTestDataloader(dataset, batch_size=self.cfg["batch_size"], shuffle=False)
        return dataloader

    def training_step(self, batch, index):
        self.train()
        device = self.cfg['device']

        (input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords,
         segment_ids_for_subwords, character_pos_to_subword_pos, labels_mask, labels, _) = batch

        input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
        input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords = (input_ids_for_subwords.to(device),
                                                                                     input_mask_for_subwords.to(device),
                                                                                     segment_ids_for_subwords.to(
                                                                                         device))
        character_pos_to_subword_pos, labels_mask, labels = (character_pos_to_subword_pos.to(device),
                                                             labels_mask.to(device), labels.to(device))

        logits = self.forward(input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords,
                              segment_ids_for_subwords, character_pos_to_subword_pos)

        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)
        loss_mask = labels_mask > 0.5
        loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
        if loss_mask_flatten.any():
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]
            loss = self.loss_function(logits_flatten, labels_flatten)
        else:
            loss = self.loss_function(logits, torch.argmax(logits, dim=-1))

        lr = self.optimizer.param_groups[0]['lr']
        print('train_loss', loss.item())
        print('lr', lr)
        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx, split="val"):
        self.eval()
        device = self.cfg['device']

        (input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords,
         character_pos_to_subword_pos, labels_mask, labels, spans) = batch

        input_ids, input_mask, segment_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device)
        input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords = input_ids_for_subwords.to(
            device), input_mask_for_subwords.to(device), segment_ids_for_subwords.to(device)
        character_pos_to_subword_pos, labels_mask, labels = character_pos_to_subword_pos.to(device), labels_mask.to(
            device), labels.to(device)

        logits = self.forward(input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords,
                              segment_ids_for_subwords, character_pos_to_subword_pos)

        tag_preds = torch.argmax(logits, dim=2)
        for input_mask_seq, segment_seq, prediction_seq, label_seq, span_seq in zip(input_mask.tolist(),
                                                                                    segment_ids.tolist(),
                                                                                    tag_preds.tolist(), labels.tolist(),
                                                                                    spans.tolist()):
            # in targets, '0' means class zero, '1' means candidate class, '2' means ending class.
            targets = []
            predictions = []

            # loop through characters with label 0
            for i in range(len(segment_seq)):
                # Skip masked values
                if input_mask_seq[i] == 0:
                    continue
                # Stop if candidate labels reached
                if segment_seq[i] > 0:
                    break
                # Check if class 0 was predicted correctly
                if label_seq[i] == 0:
                    targets.append(0)
                    if prediction_seq[i] == 0:
                        predictions.append(True)
                    else:
                        predictions.append(False)
                # Check if ending class was predicted correctly
                elif label_seq[i] > 10:
                    targets.append(2)
                    if prediction_seq[i] == label_seq[i]:
                        predictions.append(True)
                    else:
                        predictions.append(False)

            # Go over candidates and check their predictions
            for start, end in span_seq:
                if start == -1:
                    break
                targets.append(1)
                candidate_pred = prediction_seq[start:end]
                candidate_trg = label_seq[start:end]

                # cut off ending part, they are valued separately
                while candidate_trg and candidate_trg[-1] > 10:
                    candidate_trg.pop()
                    candidate_pred.pop()

                predictions.append(candidate_pred == candidate_trg)

            if len(predictions) != len(targets):
                raise ValueError(
                    "Length mismatch: len(span_labels)="
                    + str(len(targets))
                    + "; len(span_predictions)="
                    + str(len(predictions))
                )

            self.save_accuracy_vals(predictions, targets)
        self.save_batch_accuracy()

        logits_flatten = torch.flatten(logits, start_dim=0, end_dim=-2)
        labels_flatten = torch.flatten(labels, start_dim=0, end_dim=-1)
        loss_mask = labels_mask > 0.5
        loss_mask_flatten = torch.flatten(loss_mask, start_dim=0, end_dim=-1)
        if loss_mask_flatten.any():
            logits_flatten = logits_flatten[loss_mask_flatten]
            labels_flatten = labels_flatten[loss_mask_flatten]
            loss = self.loss_function(logits_flatten, labels_flatten)
        else:
            loss = self.loss_function(logits, torch.argmax(logits, dim=-1))

        if split == 'val':
            self.validation_step_outputs.append({f'{split}_loss': loss.item()})
        elif split == 'test':
            self.test_step_outputs.append({f'{split}_loss': loss.item()})

        return {f'{split}_loss': loss}

    def train_fn(self, data_loader):
        train_loss = 0.0
        for index, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
            res = self.training_step(batch, index)
            loss = res['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        return train_loss

    @torch.no_grad()
    def infer(self, input_name, output_name) -> None:
        initial_mode = self.training
        self.eval()
        device = self.cfg['device']
        self.to(device)

        try:
            dataloader = self.make_infer_data(input_name)
            tag_preds_by_sentence = []
            possible_replacements_by_sentence = []

            for batch in tqdm(dataloader, total=len(dataloader)):
                (input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords,
                 segment_ids_for_subwords, character_pos_to_subword_pos, fragment_indices) = batch

                logits = self.forward(
                    input_ids=input_ids.to(device),
                    input_mask=input_mask.to(device),
                    segment_ids=segment_ids.to(device),
                    input_ids_for_subwords=input_ids_for_subwords.to(device),
                    input_mask_for_subwords=input_mask_for_subwords.to(device),
                    segment_ids_for_subwords=segment_ids_for_subwords.to(device),
                    character_pos_to_subword_pos=character_pos_to_subword_pos.to(device),
                )

                fragments_len = fragment_indices.shape[1]
                # add row of zeroes for cumsum
                padded_logits = torch.nn.functional.pad(logits, pad=(0, 0, 1, 0))
                (batch_size, seq_len, num_labels) = padded_logits.shape
                cumsum = padded_logits.cumsum(dim=1)
                cumsum_view = cumsum.view(-1, num_labels)
                candidate_offsets = (
                        torch.ones((batch_size, fragments_len), dtype=torch.long)
                        * torch.arange(batch_size).reshape((-1, 1))
                        * seq_len
                ).view(-1)
                start_indices = (fragment_indices[..., 0]).view(-1) + candidate_offsets
                end_indices = (fragment_indices[..., 1]).view(-1) + candidate_offsets
                candidate_lengths = (end_indices - start_indices).reshape((-1, 1)).to(device)
                candidate_sums = cumsum_view[end_indices, :] - cumsum_view[start_indices, :]

                candidate_logits = (candidate_sums / candidate_lengths.float()).view(batch_size, fragments_len, num_labels)
                candidate_tag_probs = torch.nn.functional.softmax(candidate_logits, dim=-1).to(device)
                candidate_numbers = fragment_indices[:, :, 2].to(device)
                candidate_probs = torch.take_along_dim(candidate_tag_probs, candidate_numbers.long().unsqueeze(2), dim=-1).squeeze(2)
                for i in range(batch_size):
                    possible_replacements = []
                    for j in range(fragments_len):
                        start, end, candidate_id = map(int, fragment_indices[i][j])
                        # skip padding
                        if candidate_id == 0:
                            continue
                        prob = round(float(candidate_probs[i][j]), 5)
                        if prob < 0.01:
                            continue
                        possible_replacements.append(f"{start - 1} {end - 1} {candidate_id} {prob}")
                    possible_replacements_by_sentence.append(possible_replacements)

                character_preds = self.tensor2list(torch.argmax(logits, dim=-1))
                tag_preds_by_sentence.extend(character_preds)

            if len(possible_replacements_by_sentence) != len(tag_preds_by_sentence) or len(
                    possible_replacements_by_sentence) != len(
                    dataloader.get_examples()
            ):
                raise IndexError(
                    f"number of sentences differs: len(all_possible_replacements)={len(possible_replacements_by_sentence)}; "
                    f"len(all_tag_preds)={len(tag_preds_by_sentence)}; "
                    f"len(infer_datalayer.dataset.examples)={len(dataloader.get_examples())}"
                )

            with open(output_name, "w", encoding="utf-8") as out:
                for i in range(len(dataloader.get_examples())):
                    hyp, ref = dataloader.get_hypotheses()[i]
                    num_letters = hyp.count(" ") + 1
                    tag_preds_str = " ".join(map(str, tag_preds_by_sentence[i][1:num_letters + 1]))
                    possible_replacements_str = ";".join(possible_replacements_by_sentence[i])
                    out.write(f"{hyp}\t{ref}\t{possible_replacements_str}\t{tag_preds_str}\n")

        except Exception as e:
            raise ValueError(f"Error processing file {input_name}: {e}")

        finally:
            self.train(initial_mode)

    def eval_fn(self, data_loader):
        eval_loss = 0.0
        with torch.no_grad():
            for index, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
                res = self.validation_step(batch, index, 'val')
                loss = res['val_loss']
                eval_loss += loss.item()

        return eval_loss

    def run_training(self, epoch, train_data, valid_data):
        best_eval_loss = float('inf')
        for i in range(epoch):
            train_loss = self.train_fn(train_data)
            eval_loss = self.eval_fn(valid_data)

            print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                print("Saving the model")
                torch.save(self.state_dict(), self.cfg['model_name'])
                torch.save(self.optimizer.state_dict(), self.cfg['optimizer_state_path'])
            self.write_batch_accuracy("batch_accuracies")

    def save_accuracy_vals(self, predictions, targets):
        class_zero_count = 0
        class_zero_correct = 0
        candidate_count = 0
        candidates_correct = 0
        ending_count = 0
        endings_correct = 0
        for i in range(len(targets)):
            target_type = targets[i]
            pred = predictions[i]
            if target_type == 0:
                class_zero_count += 1
                if pred:
                    class_zero_correct += 1
            elif target_type == 1:
                candidate_count += 1
                if pred:
                    candidates_correct += 1
            elif target_type == 2:
                ending_count += 1
                if pred:
                    endings_correct += 1

        self.validation_acc_vals[0] += class_zero_correct
        self.validation_acc_vals[1] += class_zero_count
        self.validation_acc_vals[2] += candidates_correct
        self.validation_acc_vals[3] += candidate_count
        self.validation_acc_vals[4] += endings_correct
        self.validation_acc_vals[5] += ending_count

    def save_batch_accuracy(self):
        z1 = self.validation_acc_vals[0]
        z2 = self.validation_acc_vals[1]
        c1 = self.validation_acc_vals[2]
        c2 = self.validation_acc_vals[3]
        e1 = self.validation_acc_vals[4]
        e2 = self.validation_acc_vals[5]
        if z2 == 0:
            zacc = 1
        else:
            zacc = z1 / z2
        if c2 == 0:
            cacc = 1
        else:
            cacc = c1 / c2
        if e2 == 0:
            eacc = 1
        else:
            eacc = e1 / e2
        self.batch_accuracy.append((zacc, cacc, eacc))
        self.validation_acc_vals = [0] * 6

    def write_batch_accuracy(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            for batch_acc in self.batch_accuracy:
                zero, cand, end = batch_acc
                file.write(str(zero) + "\t" + str(cand) + "\t" + str(end) + "\n")
