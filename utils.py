# utility functions


# from a dictionary containing image information, get annotations
def list_annotations(img_info : list) -> list:
    """
    Takes the list of dictionaries returned by nsd_access.read_image_coco_info()
    and extracts just the annotations.

    Returns:
        list [annotations]
    """

    annotations = []
    
    # if we got info on more than 1 image it will be a list of lists
    if any(isinstance(el, list) for el in img_info):
        # loop through list
        for i in range(0, len(img_info)):
            img_annotations = []
            for j in range(0, len(img_info[i])):
                img_annotations.append(img_info[i][j]['caption'])
            annotations.append(img_annotations)
    else:
        for i in range(0, len(img_info)):
            annotations.append(img_info[i]['caption']) 

    return annotations
