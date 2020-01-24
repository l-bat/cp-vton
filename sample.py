import cv2 as cv
import json
import numpy as np
import os

from segmentation_net import parse_human

print(cv.__version__)

data_dir = os.path.join('data', 'test')
cloth_dir = 'cloth'
pose_dir = 'pose'
image_dir = 'image'
segm_dir = 'image-parse'


def get_cloth(cloth_name='015392_1.jpg', cloth_dir=cloth_dir):
    cloth = cv.imread(os.path.join(data_dir, cloth_dir, cloth_name))
    c = cv.dnn.blobFromImage(cloth, 1.0 / 127.5, (cloth.shape[1], cloth.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    return c

# def get_segmentation(image_name='000074_0.jpg', segm_dir=segm_dir, model_path='/home/liubov/course_work/segmentation/LIP_JPPNet/out_shape_384.pb'):
#     # image_path = os.path.join(data_dir, image_dir, image_name)
#     # segm_image = parse_human(image_path, model_path)
#     segm_path = os.path.join(data_dir, segm_dir, image_name.split('.')[0] + '.png')
#     ref = cv.imread(segm_path)
#     # cv.imshow("segm", ref )
#     # cv.imshow("segm", np.hstack((ref, segm_image)) )
#     return ref

def get_segmentation(image_name='000074_0.jpg', segm_dir=segm_dir, model_path='/home/liubov/course_work/segmentation/LIP_JPPNet/out_shape_384.pb'):
    image_path = os.path.join(data_dir, image_dir, image_name)
    segm_image = parse_human(image_path, model_path)
    print(segm_image.shape)
    return segm_image

def get_src_image(image_name='000074_0.jpg', image_dir=image_dir):
    src_image = cv.imread(os.path.join(data_dir, image_dir, image_name))
    width = src_image.shape[1]
    height = src_image.shape[0]
    src_image = cv.dnn.blobFromImage(src_image, 1.0 / 127.5, (width, height), (127.5, 127.5, 127.5), True, crop=False)
    src_image = src_image.squeeze(0)
    return src_image

def get_masks(segm_image, src_image):
    # imread
    palette = {
        'Background'   : (0, 0, 0),
        'Hat'          : (128, 0, 0),
        'Hair'         : (254, 0, 0),
        'Glove'        : (0, 85, 0),
        'Sunglasses'   : (169, 0, 51),
        'UpperClothes' : (254, 85, 0),
        'Dress'        : (0, 0, 85),
        'Coat'         : (0, 119, 220),
        'Socks'        : (85,85, 0),
        'Pants'        : (0, 85, 85),
        'Jumpsuits'    : (85, 51, 0),
        'Scarf'        : (52, 86, 128),
        'Skirt'        : (0, 128, 0),
        'Face'         : (0, 0, 254),
        'Left-arm'     : (51, 169, 220),
        'Right-arm'    : (0, 254, 254),
        'Left-leg'     : (85, 254, 169),
        'Right-leg'    : (169, 254,85),
        'Left-shoe'    : (254, 254, 0),
        'Right-shoe'   : (254, 169, 0)
    }
    # parse humangit 
    # palette = {
    #     'Background'   : (0, 0, 0),
    #     'Hat'          : (128, 0, 0),
    #     'Hair'         : (255, 0, 0),
    #     'Glove'        : (0, 85, 0),
    #     'Sunglasses'   : (170, 0, 51),
    #     'UpperClothes' : (255, 85, 0),
    #     'Dress'        : (0, 0, 85),
    #     'Coat'         : (0, 119, 221),
    #     'Socks'        : (85, 85, 0),
    #     'Pants'        : (0, 85, 85),
    #     'Jumpsuits'    : (85, 51, 0),
    #     'Scarf'        : (52, 86, 128),
    #     'Skirt'        : (0, 128, 0),
    #     'Face'         : (0, 0, 255),
    #     'Left-arm'     : (51, 170, 221),
    #     'Right-arm'    : (0, 255, 255),
    #     'Left-leg'     : (85, 255, 170),
    #     'Right-leg'    : (170, 255, 85),
    #     'Left-shoe'    : (255, 255, 0),
    #     'Right-shoe'   : (255, 170, 0)
    # }
    color2label = {val: key for key, val in palette.items()}
    head_labels = ['Hat', 'Hair', 'Sunglasses', 'Face']

    segm_image = cv.cvtColor(segm_image, cv.COLOR_BGR2RGB)
    
    width = segm_image.shape[1]
    height = segm_image.shape[0]

    phead = np.zeros((1, height, width))
    pose_shape = np.zeros((height, width, 1))
    for r in range(height):
        for c in range(width):
            pixel = segm_image[r, c]
            if tuple(pixel) in color2label:
                if color2label[tuple(pixel)] in head_labels:
                    phead[0, r, c] = 1
                if color2label[tuple(pixel)] != 'Background':
                    pose_shape[r, c, 0] = 255

    phead = phead.astype(np.float32)
    img_head = src_image * phead - (1 - phead)

    from PIL import Image
    pose_shape = pose_shape.astype(np.uint8).reshape(pose_shape.shape[0], pose_shape.shape[1])

    parse_shape = Image.fromarray(pose_shape)
    parse_shape = parse_shape.resize((width // 16, height // 16), Image.BILINEAR)
    parse_shape = parse_shape.resize((width, height), Image.BILINEAR)
    res_shape = np.array(parse_shape)

    res_shape = cv.dnn.blobFromImage(res_shape, 1.0 / 127.5, (res_shape.shape[1], res_shape.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    res_shape = res_shape.squeeze(0)
    return res_shape, img_head

def get_pose_map(height, width, radius, image_name='000074_0.jpg', image_dir=image_dir, proto_path='/home/liubov/work_spase/opencv_extra/testdata/dnn/openpose_pose_coco.prototxt'):
    image_path = os.path.join(data_dir, image_dir, image_name)
    net = cv.dnn.readNet(proto_path, proto_path.split('.')[0] + '.caffemodel')
    # net.setPreferableBackend(backend)
    # net.setPreferableTarget(target)
    img = cv.imread(image_path)
    inp = cv.dnn.blobFromImage(img, 1.0/255, (width, height), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()

    threshold = 0.1
    pose_map = np.zeros((height, width, out.shape[1] - 1))
    # last label: Background
    for i in range(0, out.shape[1] - 1):
        heatMap = out[0, i, :, :]
        keypoint = np.zeros((height, width, 1))
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (width * point[0]) / out.shape[3]
        y = (height * point[1]) / out.shape[2]
        x, y = int(x), int(y)
        if conf > threshold and x > 0 and y > 0:
            cv.rectangle(keypoint, (x - radius, y - radius), (x + radius, y + radius), (255, 255, 255), cv.FILLED)
        keypoint[:, :, 0] = (keypoint[:, :, 0] - 127.5) / 127.5
        pose_map[:, :, i] = keypoint.reshape(height, width)

    pose_map = pose_map.transpose(2, 0, 1)
    return pose_map

def prepare_agnostic(res_shape, img_head, pose_map):
    agnostic = np.concatenate((res_shape, img_head, pose_map), axis=0)
    agnostic = np.expand_dims(agnostic, axis=0)
    return agnostic

def run_gmm(agnostic, c, model_name='gmm_22_01.onnx'):
    class CorrelationLayer(object):
        def __init__(self, params, blobs):
            super(CorrelationLayer, self).__init__()

        def getMemoryShapes(self, inputs):
            fetureAShape = inputs[0]
            b, c, h, w = fetureAShape
            return [[b, h * w, h, w]]

        def forward(self, inputs):
            feature_A, feature_B = inputs
            b, c, h, w = feature_A.shape
            feature_A = feature_A.transpose(0, 1, 3, 2)
            feature_A = np.reshape(feature_A, (b, c, h * w))
            feature_B = np.reshape(feature_B, (b, c, h * w))
            feature_B = feature_B.transpose(0, 2, 1)
            feature_mul = feature_B @ feature_A
            feature_mul= np.reshape(feature_mul, (b, h, w, h * w))
            feature_mul = feature_mul.transpose(0, 1, 3, 2)
            correlation_tensor = feature_mul.transpose(0, 2, 1, 3)
            correlation_tensor = np.ascontiguousarray(correlation_tensor)
            return [correlation_tensor]

    cv.dnn_registerLayer('Correlation', CorrelationLayer)
    net = cv.dnn.readNet(model_name)

    net.setInput(agnostic, "input.1")
    net.setInput(c, "input.18")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    theta = net.forward()

    cv.dnn_unregisterLayer('Correlation')
    return theta

def get_warped_cloth(c, theta):
    grid = postprocess(theta)
    warped_cloth = bilinear_sampler(c, grid)
    return warped_cloth


from numpy.linalg import inv, det
def compute_L_inverse(X, Y):
    N = X.shape[0]

    Xmat = np.tile(X, (1, N))
    Ymat = np.tile(Y, (1, N))
    P_dist_squared = np.power(Xmat - Xmat.transpose(1, 0), 2) + np.power(Ymat - Ymat.transpose(1, 0), 2)

    P_dist_squared[P_dist_squared == 0] = 1
    K = np.multiply(P_dist_squared, np.log(P_dist_squared))

    O = np.ones([N, 1], dtype=np.float32)
    Z = np.zeros([3, 3], dtype=np.float32)
    P = np.concatenate([O, X, Y], axis=1)
    first = np.concatenate((K, P), axis=1)
    second = np.concatenate((P.transpose(1, 0), Z), axis=1)
    L = np.concatenate((first, second), axis=0)
    Li = inv(L)
    return Li

def prepare_to_transform(out_h=256, out_w=192, grid_size=5):
    grid = np.zeros([out_h, out_w, 3], dtype=np.float32)
    grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
    grid_X = np.expand_dims(np.expand_dims(grid_X, axis=0), axis=3)
    grid_Y = np.expand_dims(np.expand_dims(grid_Y, axis=0), axis=3)

    axis_coords = np.linspace(-1, 1, grid_size)
    N = grid_size ** 2
    P_Y, P_X = np.meshgrid(axis_coords, axis_coords)

    P_X = np.reshape(P_X,(-1, 1))
    P_Y = np.reshape(P_Y,(-1, 1))

    P_X = np.expand_dims(np.expand_dims(np.expand_dims(P_X, axis=2), axis=3), axis=4).transpose(4, 1, 2, 3, 0)
    P_Y = np.expand_dims(np.expand_dims(np.expand_dims(P_Y, axis=2), axis=3), axis=4).transpose(4, 1, 2, 3, 0)
    return grid_X, grid_Y, N, P_X, P_Y

def expand_torch(X, shape):
    if len(X.shape) != len(shape):
        return X.flatten().reshape(shape)
    else:
        axis = [1 if src == dst else dst for src, dst in zip(X.shape, shape)]
        return np.tile(X, axis)

def apply_transformation(theta, points, N, P_X, P_Y):
    if len(theta.shape) == 2:
        theta = np.expand_dims(np.expand_dims(theta, axis=2), axis=3)

    batch_size = theta.shape[0]

    P_X_base = np.copy(P_X)
    P_Y_base = np.copy(P_Y)

    Li = compute_L_inverse(np.reshape(P_X, (N, -1)), np.reshape(P_Y, (N, -1)))
    Li = np.expand_dims(Li, axis=0)

    # split theta into point coordinates
    Q_X = np.squeeze(theta[:, :N, :, :], axis=3)
    Q_Y = np.squeeze(theta[:, N:, :, :], axis=3)

    Q_X = Q_X + expand_torch(P_X_base, Q_X.shape)
    Q_Y = Q_Y + expand_torch(P_Y_base, Q_Y.shape)

    points_b = points.shape[0]
    points_h = points.shape[1]
    points_w = points.shape[2]

    P_X = expand_torch(P_X, (1, points_h, points_w, 1, N))
    P_Y = expand_torch(P_Y, (1, points_h, points_w, 1, N))

    W_X = expand_torch(Li[:,:N,:N], (batch_size, N, N)) @ Q_X
    W_Y = expand_torch(Li[:,:N,:N], (batch_size, N, N)) @ Q_Y

    W_X = np.expand_dims(np.expand_dims(W_X, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
    W_X = np.repeat(W_X, points_h, axis=1)
    W_X = np.repeat(W_X, points_w, axis=2)

    W_Y = np.expand_dims(np.expand_dims(W_Y, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
    W_Y = np.repeat(W_Y, points_h, axis=1)
    W_Y = np.repeat(W_Y, points_w, axis=2)

    A_X = expand_torch(Li[:, N:, :N], (batch_size, 3, N)) @ Q_X
    A_Y = expand_torch(Li[:, N:, :N], (batch_size, 3, N)) @ Q_Y

    A_X = np.expand_dims(np.expand_dims(A_X, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
    A_X = np.repeat(A_X, points_h, axis=1)
    A_X = np.repeat(A_X, points_w, axis=2)

    A_Y = np.expand_dims(np.expand_dims(A_Y, axis=3), axis=4).transpose(0, 4, 2, 3, 1)
    A_Y = np.repeat(A_Y, points_h, axis=1)
    A_Y = np.repeat(A_Y, points_w, axis=2)

    points_X_for_summation = np.expand_dims(np.expand_dims(points[:, :, :, 0], axis=3), axis=4)
    points_X_for_summation = expand_torch(points_X_for_summation, points[:, :, :, 0].shape + (1, N))

    points_Y_for_summation = np.expand_dims(np.expand_dims(points[:, :, :, 1], axis=3), axis=4)
    points_Y_for_summation = expand_torch(points_Y_for_summation, points[:, :, :, 0].shape + (1, N))

    if points_b == 1:
        delta_X = points_X_for_summation - P_X
        delta_Y = points_Y_for_summation - P_Y
    else:
        delta_X = points_X_for_summation - expand_torch(P_X, points_X_for_summation.shape)
        delta_Y = points_Y_for_summation - expand_torch(P_Y, points_Y_for_summation.shape)

    dist_squared = np.power(delta_X, 2) + np.power(delta_Y, 2)
    dist_squared[dist_squared == 0] = 1
    U = np.multiply(dist_squared, np.log(dist_squared))

    points_X_batch = np.expand_dims(points[:,:,:,0], axis=3)
    points_Y_batch = np.expand_dims(points[:,:,:,1], axis=3)

    if points_b == 1:
        points_X_batch = expand_torch(points_X_batch, (batch_size, ) + points_X_batch.shape[1:])
        points_Y_batch = expand_torch(points_Y_batch, (batch_size, ) + points_Y_batch.shape[1:])

    points_X_prime = A_X[:,:,:,:,0]+ \
                    np.multiply(A_X[:,:,:,:,1], points_X_batch) + \
                    np.multiply(A_X[:,:,:,:,2], points_Y_batch) + \
                    np.sum(np.multiply(W_X, expand_torch(U, W_X.shape)), 4)

    points_Y_prime = A_Y[:,:,:,:,0]+ \
                    np.multiply(A_Y[:,:,:,:,1], points_X_batch) + \
                    np.multiply(A_Y[:,:,:,:,2], points_Y_batch) + \
                    np.sum(np.multiply(W_Y, expand_torch(U, W_Y.shape)), 4)

    return np.concatenate((points_X_prime, points_Y_prime), 3)

def postprocess(theta):
    grid_X, grid_Y, N, P_X, P_Y = prepare_to_transform()
    warped_grid = apply_transformation(theta, np.concatenate((grid_X, grid_Y), axis=3), N, P_X, P_Y)
    return warped_grid

def bilinear_sampler(img, grid):
    x, y = grid[:,:,:,0], grid[:,:,:,1]

    H = img.shape[2]
    W = img.shape[3]
    max_y = H - 1
    max_x = W - 1

    # rescale x and y to [0, W-1/H-1]
    x = 0.5 * (x + 1.0) * (max_x - 1)
    y = 0.5 * (y + 1.0) * (max_y - 1)

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y  - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y  - y0)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = np.clip(x0, 0, max_x)
    x1 = np.clip(x1, 0, max_x)
    y0 = np.clip(y0, 0, max_y)
    y1 = np.clip(y1, 0, max_y)

    # get pixel value at corner coords
    img = img.reshape(-1, H, W)
    Ia = img[:, y0, x0].swapaxes(0, 1)
    Ib = img[:, y1, x0].swapaxes(0, 1)
    Ic = img[:, y0, x1].swapaxes(0, 1)
    Id = img[:, y1, x1].swapaxes(0, 1)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=0)
    wb = np.expand_dims(wb, axis=0)
    wc = np.expand_dims(wc, axis=0)
    wd = np.expand_dims(wd, axis=0)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    return out


def run_tom(agnostic, warp_cloth, model_name='tom_2020.onnx'):
    net = cv.dnn.readNet(model_name)
    inp = np.concatenate([agnostic, warp_cloth], axis=1)

    net.setInput(inp)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    out = net.forward()

    p_rendered, m_composite = np.split(out, [3], axis=1)
    p_rendered = np.tanh(p_rendered)

    from scipy.special import expit as sigmoid
    m_composite = sigmoid(m_composite)

    p_tryon = warp_cloth * m_composite + p_rendered * (1 - m_composite)

    rgb_p_tryon = cv.cvtColor(p_tryon.squeeze(0).transpose(1, 2, 0), cv.COLOR_BGR2RGB)
    return rgb_p_tryon


def get_pair(pair_name='test_pairs.txt'):
    path = os.path.join('data', pair_name)
    with open(path) as fin:
        for line in fin:
            image_path, cloth_path = line.split()
            print(image_path, cloth_path)
            # get_segmentation(image_path)
            test_net(image_path, cloth_path)
            # break


def test_net(image_path, cloth_path):
    height = 256
    width = 192
    radius = 5

    cloth = get_cloth(cloth_path)

    pose = get_pose_map(height, width, radius, image_path)
    segm_image = get_segmentation(image_path)
    inp_image = get_src_image(image_path)
    shape_mask, head_mask = get_masks(segm_image, inp_image)
    agnostic = prepare_agnostic(shape_mask, head_mask, pose)
    theta = run_gmm(agnostic, cloth)
    grid = postprocess(theta)

    warped_cloth = bilinear_sampler(cloth, grid).astype(np.float32)
    out = run_tom(agnostic, warped_cloth)

    # Visualize

    cloth = cloth.squeeze(0).transpose(1, 2, 0)
    cloth = cv.cvtColor(cloth, cv.COLOR_BGR2RGB)
    warped_cloth = warped_cloth.squeeze(0).transpose(1, 2, 0)
    warped_cloth = cv.cvtColor(warped_cloth, cv.COLOR_BGR2RGB)
    first_line = np.hstack((cloth, warped_cloth))

    shape_mask = shape_mask.transpose(1, 2, 0)
    shape_mask = cv.cvtColor(shape_mask, cv.COLOR_GRAY2RGB)
    head_mask = head_mask.transpose(1, 2, 0)
    head_mask = cv.cvtColor(head_mask, cv.COLOR_BGR2RGB)
    second_line = np.hstack((head_mask, shape_mask))

    inp_image = inp_image.transpose(1, 2, 0)
    inp_image = cv.cvtColor(inp_image, cv.COLOR_BGR2RGB)
    third_line = np.hstack((inp_image, out))

    torch_path = os.path.join('result', 'tom_final.pth_31_10', 'test', 'try-on', image_path)
    torch_out = cv.imread(torch_path)
    # torch_out = cv.cvtColor(torch_out, cv.COLOR_BGR2RGB)
    
    cv.imshow("torch_out", torch_out)
    # segm_image = cv.cvtColor(segm_image, cv.COLOR_BGR2RGB)
    # fouth_line = np.hstack((segm_image, torch_out))

    cv.imshow("OpenCV", np.vstack((first_line, second_line, third_line)))
    cv.waitKey()


if __name__ == "__main__":
    get_pair()
