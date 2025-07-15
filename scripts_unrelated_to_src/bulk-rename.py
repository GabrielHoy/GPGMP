import os

def changeExt(directory, extChangeFrom, extChangeTo):
    # Get all files in the directory
    files = os.listdir(directory)

    # Counter for renamed files
    renamed_count = 0

    # Iterate through each file
    for filename in files:

        # Skip files that don't match the extension we want to change from
        if not filename.endswith(extChangeFrom):
            continue


        # Construct full file path
        old_path = os.path.join(directory, filename)
        new_path = old_path.replace(extChangeFrom, extChangeTo)

        # Rename the file
        try:
            os.rename(old_path, new_path)
            renamed_count += 1
            print(f"Renamed: {filename} -> {filename}{extChangeTo}")
        except Exception as e:
            print(f"Error renaming {filename}: {str(e)}")

    print(f"\nTotal files renamed: {renamed_count}")

if __name__ == "__main__":
    # Directory containing the files
    directory = "src/mpf/MPFRoutines"

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
    else:
        changeExt(directory, '.c.TODO', '.cu')