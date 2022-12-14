import subprocess
import time

delimeter = "====================\n"


def extract_methods_from_files(path, number_of_files):
    bashCommand = "./method_extractor.sh " + path + " " + str(number_of_files)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)


def get_training_data():
    java_training_path = "../java-small/training/gradle"
    python_training_path = "./java-small/training/gradle"
    extract_methods_from_files(java_training_path, 400)
    time.sleep(5)
    with open(python_training_path + '/methods.txt', 'r') as training_file:
        training_dict = {'label': [], 'text': []}
        codelines = training_file.readlines()
        index = 0
        while index < len(codelines):
            method_name = codelines[index][:-1]
            index += 1
            method_code = ""
            while index < len(codelines) and codelines[index] != delimeter:
                method_code += codelines[index][:-1]
                index += 1
            index += 1
            masked_method_code = method_code.replace(method_name, '<mask>').strip(' \n\t')
            training_dict['label'].append(method_code)
            training_dict['text'].append(masked_method_code)
    return training_dict


def get_test_data():
    java_test_path = "../java-small/test"
    python_test_path = "./java-small/test"
    extract_methods_from_files(java_test_path, 100)
    time.sleep(5)
    with open(python_test_path + '/methods.txt', 'r') as training_file:
        test_dict = {'label': [], 'text': []}
        codelines = training_file.readlines()
        index = 0
        while index < len(codelines):
            method_name = codelines[index][:-1]
            index += 1
            method_code = ""
            while index < len(codelines) and codelines[index] != delimeter:
                method_code += codelines[index][:-1]
                index += 1
            index += 1
            masked_method_code = method_code.replace(method_name, '<mask>').strip(' \n\t')
            test_dict['label'].append(method_code)
            test_dict['text'].append(masked_method_code)
    return test_dict

