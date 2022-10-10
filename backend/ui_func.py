import os

from backend.singleton import singleton
gs = singleton



def getLatestGeneratedImagesFromPath():
    # get the latest images from the generated images folder
    # get the path to the generated images folder
    # generatedImagesPath = os.path.join(os.getcwd(), st.session_state['defaults'].general.sd_concepts_library_folder)
    # test path till we have defaults
    if gs.defaults.general.default_path_mode == "subfolders":
        generatedImagesPath = gs.defaults.general.outdir
    else:
        generatedImagesPath = f'{gs.defaults.general.outdir}/_batch_images'

    print(gs.defaults.general.default_path_mode)
    print(generatedImagesPath)
    os.makedirs(generatedImagesPath, exist_ok=True)
    # get all the files from the folders and subfolders
    files = []
    ext = ('jpeg', 'jpg', "png")
    # get the latest 10 images from the output folder without walking the subfolders
    for r, d, f in os.walk(generatedImagesPath):
        for file in f:
            if file.endswith(ext):
                files.append(os.path.join(r, file))
    # sort the files by date
    files.sort(reverse=True, key=os.path.getmtime)
    latest = files
    latest.reverse()

    # reverse the list so the latest images are first and truncate to
    # a reasonable number of images, 10 pages worth
    return latest #[Image.open(f) for f in latest[:100]]