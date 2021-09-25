JAVA_FILE_EXTENSION = '.java'
PYTHON_FILE_EXTENSION = '.py'
C_SHARP_FILE_EXTENSION = '.cs'

EXTENSION_MAPPING = {
    'java': JAVA_FILE_EXTENSION,
    'python': PYTHON_FILE_EXTENSION,
    'c#' : C_SHARP_FILE_EXTENSION
}

def get_extension_for_programming_language(programming_language):
    output = EXTENSION_MAPPING.get(programming_language)
    if output == '':
        output = None
    return output
