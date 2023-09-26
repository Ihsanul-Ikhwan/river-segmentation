from weight.checkfile import check_file


FILE_NAME = "best.pt"
model = check_file(FILE_NAME)

print(model)

if model is None:
    raise FileNotFoundError(f"{FILE_NAME} is not found!")
