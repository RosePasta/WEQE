
def remove_double_spaces(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
    return string