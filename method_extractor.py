import javalang
import os

def __get_start_end_for_node(tree, node_to_find):
    start = None
    end = None
    for path, node in tree:
        if start is not None and node_to_find not in path:
            end = node.position
            return start, end
        if start is None and node == node_to_find:
            start = node.position
    return start, end


def __get_text(data, start, end):
    if start is None:
        return ""

    # positions are all offset by 1. e.g. first line -> lines[0], start.line = 1
    end_pos = None

    if end is not None:
        end_pos = end.line - 1

    lines = data.splitlines(True)
    string = "".join(lines[start.line:end_pos])
    string = lines[start.line - 1] + string

    # When the method is the last one, it will contain a additional brace
    if end is None:
        left = string.count("{")
        right = string.count("}")
        if right - left == 1:
            p = string.rfind("}")
            string = string[:p]

    return string


def get_method_start_end(method_node, tree):
    startpos = None
    endpos = None
    startline = None
    endline = None
    for path, node in tree:
        if startpos is not None and method_node not in path:
            endpos = node.position
            endline = node.position.line if node.position is not None else None
            break
        if startpos is None and node == method_node:
            startpos = node.position
            startline = node.position.line if node.position is not None else None
    return startpos, endpos, startline, endline


def get_method_text(codelines, startpos, endpos, startline, endline, last_endline_index):
    if startpos is None:
        return "", None, None, None
    else:
        startline_index = startline - 1
        endline_index = endline - 1 if endpos is not None else None

        # 1. check for and fetch annotations
        if last_endline_index is not None:
            for line in codelines[(last_endline_index + 1):(startline_index)]:
                if "@" in line:
                    startline_index = startline_index - 1
        meth_text = "<ST>".join(codelines[startline_index:endline_index])
        meth_text = meth_text[:meth_text.rfind("}") + 1]

        # 2. remove trailing rbrace for last methods & any external content/comments
        # if endpos is None and 
        if not abs(meth_text.count("}") - meth_text.count("{")) == 0:
            # imbalanced braces
            brace_diff = abs(meth_text.count("}") - meth_text.count("{"))

            for _ in range(brace_diff):
                meth_text = meth_text[:meth_text.rfind("}")]
                meth_text = meth_text[:meth_text.rfind("}") + 1]

        meth_lines = meth_text.split("<ST>")
        meth_text = "".join(meth_lines)
        last_endline_index = startline_index + (len(meth_lines) - 1)

        return meth_text, (startline_index + 1), (last_endline_index + 1), last_endline_index


def extract_method_from_file(target_file):
    with open(target_file, 'r') as r:
        codelines = r.readlines()
        code_text = ''.join(codelines)

    lex = None
    tree = javalang.parse.parse(code_text)
    methods = {}
    for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
        # startpos, endpos, startline, endline = get_method_start_end(method_node, tree)
        # method_text, startline, endline, lex = get_method_text(codelines, startpos, endpos, startline, endline, lex)
        start, end = __get_start_end_for_node(tree, method_node)
        method_code = __get_text(code_text,start, end)
        methods[method_node.name] = method_code
        # print(method_text)
        # print("\nZALUPA\n")
    return methods


bad_files = [".DS_Store", "StatementAnalyzer.java", "DateTimeUtils.java", "Hamlet.java", "GenericXmlContextLoaderResourceLocationsTests.java", "GenericConversionServiceTests.java"]

def extract_methods_from_all_files():
    training_data = {}
    test_data = {}
    cnt = 0
    for root, dirs, files in os.walk("./data/java-small/training/spring-framework"):
        for file in files:
            cnt += 1
            # print(cnt)
            if cnt == 300:
                break
            if file in bad_files:
                continue
            # print(os.path.join(root, file))
            new_methods = extract_method_from_file(os.path.join(root, file))
            training_data.update(new_methods)
    cnt = 0
    for root, dirs, files in os.walk("./data/java-small/test"):
        for file in files:
            cnt += 1
            # print(cnt)
            if cnt == 50:
                break
            if file in bad_files:
                continue
            # print(os.path.join(root, file))
            new_methods = extract_method_from_file(os.path.join(root, file))
            test_data.update(new_methods)
    for method_name, method_code in training_data.items():
        method_code = method_code.replace(method_name, '<mask>')
        training_data[method_name] = method_code
    for method_name, method_code in test_data.items():
        method_code = method_code.replace(method_name, '<mask>')
        test_data[method_name] = method_code
    # print(training_data)
    return training_data, test_data


# training, test = extract_methods_from_all_files()
# for key, value in training.items():
#     print(value)
#     break