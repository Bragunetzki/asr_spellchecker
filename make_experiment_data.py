import math
import os
import shutil
import string


def count_words_in_file(file_path):
    """Count the number of words in a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()
        return len(words)


def clean_word(word):
    """Remove punctuation from a word."""
    translator = str.maketrans('', '', string.punctuation)
    return word.translate(translator)


def copy_files_with_min_words(src_directory, dest_directory, min_words=400, max_files=2000):
    """Copy files with at least min_words words from src_directory to dest_directory."""
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    copied_files_count = 0

    for filename in os.listdir(src_directory):
        if copied_files_count >= max_files:
            break

        file_path = os.path.join(src_directory, filename)
        if os.path.isfile(file_path):
            try:
                word_count = count_words_in_file(file_path)
                if word_count >= min_words:
                    shutil.copy(file_path, dest_directory)
                    copied_files_count += 1
                    print(f"Copied {filename} ({word_count} words)")
            except Exception as e:
                print(f"Could not process file {filename}. Error: {e}")

    print(f"Total files copied: {copied_files_count}")


def calculate_idf(directory):
    """Calculate IDF for each word in all files in the directory."""
    word_count = {}
    total_files = 0

    # Count occurrences of each word in all files
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                words = content.split()
                unique_words = set(clean_word(word) for word in words if word.isalpha())
                total_files += 1

                for word in unique_words:
                    word_count[word] = word_count.get(word, 0) + 1

    # Calculate IDF for each word
    idf = {}
    for word, count in word_count.items():
        idf[word] = math.log(total_files / (count + 1))  # Add 1 to avoid division by zero

    return idf


def process_file(file_path, idf, idf_threshold=2.0):
    """Process a file to find words with IDF > idf_threshold and capitalized phrases."""
    words_with_idf = []
    capitalized_phrases = []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        words = content.split()

        # Find words with IDF > idf_threshold
        for word in words:
            cleaned_word = clean_word(word)
            if cleaned_word.isalpha() and idf.get(cleaned_word, 0) > idf_threshold:
                words_with_idf.append(cleaned_word)

        # Find capitalized phrases not at the beginning of the sentence
        for i in range(len(words)):
            cleaned_word = clean_word(words[i])
            if cleaned_word and cleaned_word[0].isupper() and (i == 0 or not words[i - 1].endswith('.')):
                capitalized_phrases.append(cleaned_word)

    return words_with_idf, capitalized_phrases


def write_to_dictionary(file_path, words_with_idf, capitalized_phrases):
    """Write words with IDF > 2.0 and capitalized phrases to a separate dictionary file."""
    dictionary_file_path = file_path + "_dictionary.txt"
    with open(dictionary_file_path, 'w', encoding='utf-8') as dictionary_file:
        for word in words_with_idf:
            dictionary_file.write(word + '\n')
        for phrase in capitalized_phrases:
            dictionary_file.write(phrase + '\n')


if __name__ == "__main__":
    src_directory = input("Enter the source directory path: ")
    dest_directory = input("Enter the destination directory path: ")
    copy_files_with_min_words(src_directory, dest_directory)

    idf = calculate_idf(dest_directory)

    for filename in os.listdir(dest_directory):
        file_path = os.path.join(dest_directory, filename)
        if os.path.isfile(file_path):
            words_with_idf, capitalized_phrases = process_file(file_path, idf, 2.5)
            write_to_dictionary(file_path, words_with_idf, capitalized_phrases)
            print(f"Processed {file_path}")
