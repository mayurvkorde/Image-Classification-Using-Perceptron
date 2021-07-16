dir = r'E:\sklearn Datasets\oily'
for count, filename in enumerate(os.listdir(dir)):
    dst = os.path.join(dir, f"Oily_{count}.jpg")
    src = os.path.join(dir, filename)
 
    os.rename(src, dst)
