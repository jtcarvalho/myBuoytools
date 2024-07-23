import os

def get_file_names(directory, prefix_to_remove=""):
    """Returns a set of file names in the directory without the given prefix."""
    file_names = set()
    for file_name in os.listdir(directory):
        if prefix_to_remove and file_name.startswith(prefix_to_remove):
            file_name = file_name[len(prefix_to_remove):]
        file_names.add(file_name)
    return file_names

def compare_directories(dir1, dir2, prefix):
    # Obter os nomes dos arquivos em cada diretório
    dir1_files = get_file_names(dir1)
    dir2_files_with_prefix = set(os.listdir(dir2))  # Arquivos com prefixo no dir2
    dir2_files = get_file_names(dir2, prefix)  # Arquivos sem prefixo no dir2
    
    # Contar os arquivos em cada diretório
    count_dir1 = len(dir1_files)
    count_dir2 = len(dir2_files)
    
    # Identificar arquivos que não são iguais
    only_in_dir1 = dir1_files - dir2_files
    only_in_dir2 = dir2_files - dir1_files
    
    return count_dir1, count_dir2, only_in_dir1, only_in_dir2, dir2_files_with_prefix

def remove_files(directory, files, prefix=""):
    """Remove files from the directory that match the names in the files set."""
    for file_name in os.listdir(directory):
        if prefix and file_name.startswith(prefix):
            original_name = file_name[len(prefix):]
        else:
            original_name = file_name
        if original_name in files:
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)
            print(f"Removido: {file_path}")

# Exemplo de uso:
dir1 = '/data/cmcc/jc11022/buoys/emodnet/'
exp = 'expb2_143_psi'
dir2 = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points/'
prefix = "ww3_"

count_dir1, count_dir2, only_in_dir1, only_in_dir2, dir2_files_with_prefix = compare_directories(dir1, dir2, prefix)

print(f"Número de arquivos em {dir1}: {count_dir1}")
print(f"Número de arquivos em {dir2}: {count_dir2}")

print("Arquivos presentes apenas em {dir1}:")
for file in only_in_dir1:
    print(file)

print("Arquivos presentes apenas em {dir2}:")
for file in only_in_dir2:
    print(file)

# Remover os arquivos do diretório 2 que não estão no diretório 1
files_to_remove = {f"{prefix}{file}" for file in only_in_dir2}
remove_files(dir2, files_to_remove, "")
# Remover os arquivos do diretório 1 que não estão no diretório 2
files_to_remove = {f"{file}" for file in only_in_dir1}
remove_files(dir1, files_to_remove, "")
