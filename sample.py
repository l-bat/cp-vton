import cv2 as cv
import json
import numpy as np

print(cv.__version__)

def prepare_cloth(path='/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/cloth/015392_1.jpg'):
    cloth = cv.imread(path)
    # cv.imshow("cloth", cloth)
    # cv.waitKey()
    c = cv.dnn.blobFromImage(cloth, 1.0 / 127.5, (cloth.shape[1], cloth.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    return c

def prepare_warp_cloth(path='/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/warp-cloth/015392_1.jpg'):
    cloth = cv.imread(path)
    # cv.imshow("cloth", cloth)
    # cv.waitKey()
    c = cv.dnn.blobFromImage(cloth, 1.0 / 127.5, (cloth.shape[1], cloth.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    return c

def prepare_agnostic(path='/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/image-parse/000074_0.png'):
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
    color2label = {val: key for key, val in palette.items()}

    segm_image = cv.imread(path)
    segm_image = cv.cvtColor(segm_image, cv.COLOR_BGR2RGB)

    head_labels = ['Hat', 'Hair', 'Sunglasses', 'Face']
    phead = np.zeros((1, segm_image.shape[0], segm_image.shape[1]))
    pose_shape = np.zeros((segm_image.shape[0], segm_image.shape[1], 1))
    for r in range(segm_image.shape[0]):
        for c in range(segm_image.shape[1]):
            pixel = segm_image[r, c]
            if tuple(pixel) in color2label:
                if color2label[tuple(pixel)] in head_labels:
                    phead[0, r, c] = 1
                if color2label[tuple(pixel)] != 'Background':
                    pose_shape[r, c, 0] = 255

    phead = phead.astype(np.float32)

    src_image = cv.imread('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/image/000074_0.jpg')
    src_image = cv.dnn.blobFromImage(src_image, 1.0 / 127.5, (src_image.shape[1], src_image.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    src_image = src_image.reshape((-1, src_image.shape[2], src_image.shape[3]))
    img_head = src_image * phead - (1 - phead)

    width = segm_image.shape[1]
    height = segm_image.shape[0]
    res_shape = cv.resize(pose_shape.astype(np.uint8), dsize=(width // 16, height // 16), interpolation=cv.INTER_LINEAR) # better INTER_CUBIC
    res_shape = cv.resize(res_shape, dsize=(width, height), interpolation=cv.INTER_LINEAR)
    
    res_shape = cv.dnn.blobFromImage(res_shape, 1.0 / 127.5, (res_shape.shape[1], res_shape.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    res_shape = res_shape.reshape((-1, res_shape.shape[2], res_shape.shape[3]))

    pose_map = get_pose_map()

    # print("res_shape", res_shape.shape)
    # print("phead", img_head.shape)
    # print("pose_map", pose_map.shape)

    agnostic = np.concatenate((res_shape, img_head, pose_map), axis=0)
    agnostic = np.expand_dims(agnostic, axis=0)
    # print("agnostic", agnostic.shape)
    return agnostic

def get_pose_map(path='/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/pose/000074_0_keypoints.json'):
    with open(path) as fin:
        pose = json.load(fin)['people'][0]['pose_keypoints']

    pose = np.array(pose).reshape((-1, 3))
    point_num = pose.shape[0]        
    pose_map = np.zeros((height, width, point_num))
    for i in range(point_num):
        keypoint = np.zeros((height, width, 1))
        x, y, _ = pose[i]
        x, y = int(x), int(y)
        if x > 0 and y > 0:
            cv.rectangle(keypoint, (x - radius, y - radius), (x + radius, y + radius), (255, 255, 255), cv.FILLED)
        keypoint[:, :, 0] = (keypoint[:, :, 0] - 127.5) / 127.5
        pose_map[:, :, i] = keypoint.reshape(height, width)

    pose_map = pose_map.swapaxes(0, 1).swapaxes(0, 2)
    # print("pose_map", np.max(pose_map), np.min(pose_map))
    # ref_pose_map = np.load('ref_pose_map.npy')
    # print("ref_pose_map", np.max(ref_pose_map), np.min(ref_pose_map))
    # print("pose diff =", np.max(abs(pose_map - ref_pose_map)))
    return pose_map


def run_gmm(agnostic, c):
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
    onnxmodel = "gmm.onnx"
    # inp1 = np.load("inp1_gmm.npy")
    # inp2 = np.load("inp2_gmm.npy")
    # ref = np.load("out_gmm_theta.npy")
    net = cv.dnn.readNet(onnxmodel)

    net.setInput(agnostic, "input.1")
    net.setInput(c, "input.18")
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    theta = net.forward()
    # ref = np.load('out_gmm_theta.npy')
    # print(np.max(abs(theta - ref)))
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
    # print("L", det(L))
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

    # print(img.shape)
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
    # print("Id", Id.shape)
    # print("wd", wd.shape)

    # add dimension for addition
    wa = np.expand_dims(wa, axis=0)
    wb = np.expand_dims(wb, axis=0)
    wc = np.expand_dims(wc, axis=0)
    wd = np.expand_dims(wd, axis=0)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id
    return out


def prepare_input_tom(height, width, radius):
    pose_data = '/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/pose/000074_0_keypoints.json'
    with open(pose_data) as fin:
        pose = json.load(fin)['people'][0]['pose_keypoints']

    pose = np.array(pose).reshape((-1, 3))
    point_num = pose.shape[0]        
    pose_map = np.zeros((height, width, point_num))
    for i in range(point_num):
        keypoint = np.zeros((height, width, 1))
        x, y, _ = pose[i]
        x, y = int(x), int(y)
        if x > 0 and y > 0:
            cv.rectangle(keypoint, (x - radius, y - radius), (x + radius, y + radius), (255, 255, 255), cv.FILLED)
        keypoint[:, :, 0] = (keypoint[:, :, 0] - 127.5) / 127.5
        pose_map[:, :, i] = keypoint.reshape(height, width)

    pose_map = pose_map.swapaxes(0, 1).swapaxes(0, 2)


    src_image = cv.imread('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/image/000074_0.jpg')
    # cv.imshow("src", src_image)
    segm_image = cv.imread('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/image-parse/000074_0.png')
    # cv.imshow("segm_image", segm_image)
    print("segm_image",  np.max(segm_image), np.min(segm_image))


    phead = (segm_one == 1).astype(np.float32) + (segm_one == 2).astype(np.float32) + (segm_one == 4).astype(np.float32) + (segm_one == 13).astype(np.float32) # hat, hair, sunglasses, face

    # phead = (segm_image == 1).astype(np.float32) + (segm_image == 2).astype(np.float32) + (segm_image == 4).astype(np.float32) + (segm_image == 13).astype(np.float32) # hat, hair, sunglasses, face
    ref_parse_head = np.load('ref_parse_head.npy')
    print("phead",  np.max(phead), np.min(phead))
    print("ref_parse_head",  np.max(ref_parse_head), np.min(ref_parse_head))
    print("HEAD diff =", np.max(abs(ref_parse_head - phead)))

    # img_head = src_image * phead - (1 - phead)
    # img_head = img_head.swapaxes(0, 1).swapaxes(0, 2)
    # ref_im_h = np.load('ref_im_h.npy')
    # print("ref_im_h diff =", np.max(abs(ref_im_h - img_head)))

    # p = (segm_image[:, :, 0] + segm_image[:, :, 1] + segm_image[:, :, 2]) / 3
    parse_shape = (segm_image > 0).astype(np.float32) # [0, 1]
    parse_shape = (parse_shape * 255).astype(np.uint8)
    print("parse_shape", np.min(parse_shape), np.max(parse_shape))
    from PIL import Image
    refer = Image.fromarray(parse_shape)
    refer = refer.resize((width//16, height//16), Image.BILINEAR)
    refer = refer.resize((width, height), Image.BILINEAR)
    # refer.show()
    # print("refer", np.min(refer), np.max(refer))


    parse_shape = cv.resize(parse_shape, dsize=(width // 16, height // 16), interpolation=cv.INTER_LINEAR)
    parse_shape = cv.resize(parse_shape, dsize=(width, height), interpolation=cv.INTER_LINEAR)
    # cv.imshow("ocv", cv.cvtColor(parse_shape, cv.COLOR_BGR2RGB))
    # cv.waitKey()
    # print("parse_shape", parse_shape)
    # print("parse_shape", np.min(parse_shape), np.max(parse_shape))
    print("DIFFFF", np.max(abs(np.array(refer) - cv.cvtColor(parse_shape, cv.COLOR_BGR2RGB))))

    parse_shape = cv.dnn.blobFromImage(parse_shape, 1.0 / 127.5, (parse_shape.shape[1], parse_shape.shape[0]), (127.5, 127.5, 127.5), True, crop=False) # do we need 4 dims? 
   
    parse_shape = parse_shape[:, 0, :, :]
    parse_shape = parse_shape.reshape(-1, height, width)


    ref_shape = np.load('ref_shape.npy')
    print("SHAPE diff =", np.max(abs(ref_shape - parse_shape)))

    agnostic = np.concatenate((parse_shape, img_head, pose_map), axis=0)
    agnostic = np.expand_dims(agnostic, axis=0)
    
    # ref_agnostic = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/ref_agnostic.npy')
    # print("max diff agnostic =", np.max(abs(ref_agnostic - agnostic)))


    # warp_cloth = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/before_c.npy')
    warp_cloth = cv.imread('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/data/test/warp-cloth/015392_1.jpg')
    # cv.imshow("warp_cloth", warp_cloth)
    # cv.waitKey()
    c = cv.dnn.blobFromImage(warp_cloth, 1.0 / 127.5, (warp_cloth.shape[1], warp_cloth.shape[0]), (127.5, 127.5, 127.5), True, crop=False)
    # c = c.squeeze(0)
    # after_c = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/after_c.npy')
    # print("max diff c =", np.max(abs(c - after_c)))


    # print("max diff =", np.max(abs(after_c - c)))

    # ref_c = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/test_c.npy')

    # cv.imshow("origin", cv.cvtColor(ref_c.squeeze(0).swapaxes(0, 1).swapaxes(1, 2), cv.COLOR_BGR2RGB))
    # cv.imshow("inp", warp_cloth)
    # cv.imshow("c", cv.cvtColor(c.squeeze(0).swapaxes(0, 1).swapaxes(1, 2).astype(np.uint8), cv.COLOR_BGR2RGB))
    # cv.imshow("ref", cv.cvtColor(after_c.swapaxes(0, 1).swapaxes(1, 2), cv.COLOR_BGR2RGB))
    # # # print("max diff =", np.max(abs(p_tryon - torch_res)))
    # cv.waitKey()



    # print("max diff c =", np.max(abs(c - ref_c)))

    # print("agnostic", agnostic.shape)
    # print("c", c.shape)
    # inp_ref = np.load("inp_tom.npy")
    # print("inp_ref", inp_ref.shape)
    # inp = np.concatenate([agnostic, c], axis=1)
    # print("max diff agnostic =", np.max(abs(inp_ref - inp)))

    return agnostic, c


def run_tom(agnostic, warp_cloth):
    onnxmodel = "tom.onnx"
    net = cv.dnn.readNet(onnxmodel)
    inp = np.concatenate([agnostic, warp_cloth], axis=1)
    # inp = np.load("inp_tom.npy")

    net.setInput(inp)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    out = net.forward()

    p_rendered, m_composite = np.split(out, [3], axis=1)
    p_rendered = np.tanh(p_rendered)

    from scipy.special import expit as sigmoid
    m_composite = sigmoid(m_composite)

    # c = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/c.npy')
    # print("C", c.shape)
    p_tryon = warp_cloth * m_composite + p_rendered * (1 - m_composite)
    # print("p", p_tryon.shape)
    torch_res = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/p_tryon.npy')

    rgb_p_tryon = cv.cvtColor(p_tryon.squeeze(0).swapaxes(0, 1).swapaxes(1, 2), cv.COLOR_BGR2RGB)
    rgb_torch_res = cv.cvtColor(torch_res.squeeze(0).swapaxes(0, 1).swapaxes(1, 2), cv.COLOR_BGR2RGB)    
    cv.imshow("opencv", rgb_p_tryon)
    cv.imshow("origin", rgb_torch_res)
    # # # print("max diff =", np.max(abs(p_tryon - torch_res)))
    cv.waitKey()
    return cv.cvtColor(p_tryon.squeeze(0).swapaxes(0, 1).swapaxes(1, 2), cv.COLOR_BGR2RGB)


if __name__ == "__main__":
    height = 256
    width = 192
    radius = 5

    cloth = prepare_cloth()
    agnostic = prepare_agnostic()
    theta = run_gmm(agnostic, cloth)
    grid = postprocess(theta)

    warped_cloth = bilinear_sampler(cloth, grid)
    warped_cloth = (warped_cloth - 127.5) / 127.5
    # warped_cloth[:, 0, :, :] = (warped_cloth[:, 0, :, :] - 127.5) / 127.5
    # warped_cloth[:, 1, :, :] = (warped_cloth[:, 1, :, :] - 127.5) / 127.5
    # warped_cloth[:, 2, :, :] = (warped_cloth[:, 2, :, :] - 127.5) / 127.5

    warped_cloth = warped_cloth.astype(np.float32)
    print("[", np.min(warped_cloth), np.max(warped_cloth), "]")
    # print("c", c.shape)
    # print("cloth_ref", cloth_ref.shape)
    print("warped_cloth", warped_cloth.shape)
    # print("postprocess", grid.shape)
    # warped_cloth = bilinear_sampler(cloth, grid)
    # print("warp", warped_cloth.shape)
    # warped_cloth = cv.dnn.blobFromImage(warped_cloth, 1.0 / 127.5, (warped_cloth.shape[1], warped_cloth.shape[0]), (127.5, 127.5, 127.5), True, crop=False)

    # # warped_cloth = prepare_warp_cloth()
    out = run_tom(agnostic, warped_cloth)



    # X = np.random.uniform(-1, 1, size=[25, 1])
    # Y = np.random.uniform(-1, 1, size=[25, 1])
    # Li = compute_L_inverse(X, Y)

    # theta = np.random.uniform(-1, 1, size=(1, 50))
    # theta = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/out_gmm_theta.npy')
    # grid = postprocess(theta)
    # ref = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/out_gmm_grid.npy')
    # print("grid", np.max(grid))
    # print("ref", np.max(ref))
    # print(np.max(abs(grid - ref)))

    # grid = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/grid_sample1.npy')
    # cm = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/cm_sample1.npy')
    # ref = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/warped_mask_sample1.npy')
    # # print(grid.shape)
    # # print(cm.shape)
    # print(ref.shape)
    # res = bilinear_sampler(cm, grid)
    # print("res", np.min(res))
    # print("ref", np.min(ref))
    # print("max diff =", np.max(abs(res - ref)))
    # print("ref", ref.shape)
    # print("res", res.shape)

    # # image_ref = ref.reshape(ref.shape[2], ref.shape[3], -1)
    # # cv.imshow("origin", image_ref)
    # # cv.imshow("opencv", res.reshape(ref.shape[2], ref.shape[3], -1))
    # # cv.waitKey()

    # c = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/c_sample0.npy')
    # cloth_ref = np.load('/home/liubov/course_work/cluster/user/lbatanin/cp-vton/warped_cloth_sample0.npy')
    # warped_cloth = bilinear_sampler(c, grid)
    # print("c", c.shape)
    # print("cloth_ref", cloth_ref.shape)
    # print("warped_cloth", warped_cloth.shape)
    # print("max diff =", np.max(abs(cloth_ref - warped_cloth)))

    # image_ref = cloth_ref.reshape(cloth_ref.shape[2], cloth_ref.shape[3], -1)
    # cv.imshow("origin", image_ref)
    # cv.imshow("opencv", warped_cloth.reshape(warped_cloth.shape[2], warped_cloth.shape[3], -1))
    # cv.waitKey()
