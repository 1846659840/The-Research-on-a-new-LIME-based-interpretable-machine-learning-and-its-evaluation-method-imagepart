import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.segmentation
import copy, os
import cv2
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings

print('Notebook running: keras ', keras.__version__)
np.random.seed(222)

warnings.filterwarnings('ignore')
inceptionV3_model = keras.applications.inception_v3.InceptionV3()


def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image


def click_event(event, x, y, flags, params):
    x, y = y, x

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        print(x, ' ', y, ' blook = ', superpixels[x, y], end='\r')

        if superpixels[x, y] in inactive_block:
            inactive_block.remove(superpixels[x, y])
        else:
            inactive_block.append(superpixels[x, y])

        active_pixels = list(set(active_block).difference(set(inactive_block)))

        mask = np.zeros(superpixels.shape)
        for active in active_pixels:
            mask[superpixels == active] = 1

        perturbed_image = copy.deepcopy(Xi)
        perturbed_image = perturbed_image * mask[:, :, np.newaxis]
        img = perturbed_image[:, :, ::-1]


def write_pic(Num, path_lj):
    cell = f'{Num}{imgs_num}'
    img = Image(path_lj)
    img.width, img.height = 140, 140
    sht.row_dimensions[imgs_num].height = 110
    sht.column_dimensions[Num].width = 25
    sh_img.add_image(img, cell)


source_path = 'F:\\newimage'
ALL_imgs_file = []
for file in os.listdir(source_path):
    all_files = f'{source_path}/{file}'
    if '.' in file:
        continue
    for imgs in os.listdir(all_files):
        imgs_file = f'{all_files}/{imgs}'
        ALL_imgs_file.append(imgs_file)

for imgs_file in ALL_imgs_file:
    # imgs_file ='/Users/lyra/Desktop/111/newimage/elephant/470.jpeg'
    imgs_num = imgs_file.split('/')
    imgs_num = imgs_num[-1].split('.')
    imgs_num = imgs_num[0]

    dirpath = 'newimage'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    dirpath_image = f'{dirpath}/{imgs_num}'
    if not os.path.exists(dirpath_image):
        os.mkdir(dirpath_image)

    Xi = skimage.io.imread(imgs_file)
    Xi = skimage.transform.resize(Xi, (299, 299))
    Xi = (Xi - 0.5) * 2  # Inception pre-processing
    # skimage.io.imshow(Xi/2+0.5) # Show image before inception preprocessing
    skimage.io.imsave(f'{dirpath_image}/0.jpg', Xi / 2 + 0.5)

    np.random.seed(222)
    preds = inceptionV3_model.predict(Xi[np.newaxis, :, :, :])
    decode_predictions(preds)[0]  # Top 5 classes

    top_pred_classes = preds[0].argsort()[-6:][::-1]

    from skimage.segmentation import mark_boundaries

    superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4, max_dist=200, ratio=0.2)
    skimage.io.imshow(mark_boundaries(Xi / 2 + 0.5, superpixels))
    num_superpixels = np.unique(superpixels).shape[0]

    num_perturb = 150
    perturbations = []
    for i in range(num_perturb):
      perturbations.append(np.random.choice(2,num_superpixels,p=[0.1,0.9]))

    skimage.io.imshow(perturb_image(Xi / 2 + 0.5, perturbations[0], superpixels))

    predictions = []
    for pert in perturbations:
        perturbed_img = perturb_image(Xi, pert, superpixels)
        pred = inceptionV3_model.predict(perturbed_img[np.newaxis, :, :, :])
        predictions.append(pred)
    predictions = np.array(predictions)
    predictions[0]

    active_block = np.unique(superpixels)

    # reading the image
    img = Xi[:, :, ::-1]



    inactive_block = []

    with open("C:\\Users\\Administrator\\Desktop\\pic\\butterfly.txt", "r",encoding='UTF-8') as f:  # 打开文件
        for line in f.readlines():
            line = line.split(' ')
            one_line = line[0].replace('[', '')
            one_line = one_line.replace(',', '')
            one_line = one_line.replace('\u200b', '')
            one_line = int(one_line)
            if int(imgs_num) == one_line:
                print(line)
                inactive_block = []
                for i in line[1:]:
                    i = i.replace('[', '')
                    i = i.replace(',', '')
                    i = i.replace(']', '')
                    i = i.replace('\n', '')
                    i = i.replace('\u200b', '')
                    i = int(i)
                    inactive_block.append(i)
                print(inactive_block)
                break
        f.close()

    #inactive_block.sort()

    array = np.array(inactive_block)

    original_image = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()

    distances.shape
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function

    class_to_explain = top_pred_classes[0]
    simpler_model = LinearRegression()
    simpler_model.fit(X=perturbations, y=predictions[:, :, class_to_explain], sample_weight=weights)
    coeff = simpler_model.coef_[0]
    intercept = simpler_model.intercept_[0]

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.multioutput import MultiOutputRegressor

    regr_rf = RandomForestRegressor()
    regr_rf.fit(X=perturbations, y=predictions[:, :, class_to_explain],sample_weight=weights)

    y_pred = regr_rf.predict(perturbations)
    importances = regr_rf.feature_importances_

    num_top_features = len(array)
    top_features2 = np.argsort(importances)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features2] = True  # Activate top superpixels

    images_rf = perturb_image(Xi / 2 + 0.5, mask, superpixels)
    # skimage.io.imshow(images_rf)
    skimage.io.imsave(f'{dirpath_image}/1.jpg', images_rf)

    intercept = simpler_model.intercept_[0]

    num_top_features=len(array)
    top_features = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features] = True  # Activate top superpixels

    images_lime = perturb_image(Xi / 2 + 0.5, mask, superpixels)
    # skimage.io.imshow(images_lime)
    skimage.io.imsave(f'{dirpath_image}/2.jpg', images_lime)

    rfTrue=(len(set(array) & set(top_features2))/len(array))
    limeTrue=(len(set(array) & set(top_features))/len(array))

    from sklearn import tree

    treemodel = tree.DecisionTreeRegressor(max_depth=5)
    treemodel.fit(perturbations, predictions[:, :, class_to_explain], sample_weight=weights)
    importances = treemodel.feature_importances_
    num_top_features = len(array)
    top_features3 = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features3] = True  # Activate top superpixels

    images_tree = perturb_image(Xi / 2 + 0.5, mask, superpixels)
    # skimage.io.imshow(images_tree)
    skimage.io.imsave(f'{dirpath_image}/3.jpg', images_tree)

    TreeTrue=(len(set(array) & set(top_features3))/len(array))

    import xgboost as xgb

    xgbrModel = xgb.XGBRegressor()
    xgbrModel.fit(perturbations, predictions[:, :, class_to_explain],sample_weight=weights)
    importances = xgbrModel.feature_importances_

    num_top_features = len(array)
    top_features4 = np.argsort(coeff)[-num_top_features:]

    mask = np.zeros(num_superpixels)
    mask[top_features4] = True  # Activate top superpixels

    images_xgbst = perturb_image(Xi / 2 + 0.5, mask, superpixels)
    # skimage.io.imshow(images_xgbst)
    skimage.io.imsave(f'{dirpath_image}/4.jpg', images_xgbst)

    xgbstTrue=(len(set(array) & set(top_features4))/len(array))

    from openpyxl.drawing.image import Image
    from openpyxl import load_workbook

    wb = load_workbook('F:\\All_data1.xlsx')
    sht = wb['Sheet1']
    sh_img = wb.active

    imgs_num = int(imgs_num)
    sht[f'F{imgs_num}'].value = rfTrue
    sht[f'G{imgs_num}'].value = limeTrue
    sht[f'H{imgs_num}'].value = TreeTrue
    sht[f'I{imgs_num}'].value = xgbstTrue

    write_pic(f'A', f'{dirpath_image}/0.jpg')
    write_pic(f'B', f'{dirpath_image}/1.jpg')
    write_pic(f'C', f'{dirpath_image}/2.jpg')
    write_pic(f'D', f'{dirpath_image}/3.jpg')
    write_pic(f'E', f'{dirpath_image}/4.jpg')

    print(rfTrue, limeTrue, TreeTrue, xgbstTrue)
    wb.save('F:\\All_data1.xlsx')
    wb.close()
    print('保存完毕！')