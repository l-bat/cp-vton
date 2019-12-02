import cv2 as cv
import numpy as np

print(cv.__version__)

# class CorrelationLayer(object):
#     def __init__(self, params, blobs):
#          super(CorrelationLayer, self).__init__()

#     def getMemoryShapes(self, inputs):
#         fetureAShape = inputs[0]
#         b, c, h, w = fetureAShape
#         return [[b, h * w, h, w]]

#     def forward(self, inputs):
#         feature_A, feature_B = inputs
#         b, c, h, w = feature_A.shape
#         feature_A = feature_A.transpose(0, 1, 3, 2)
#         feature_A = np.reshape(feature_A, (b, c, h * w))
#         feature_B = np.reshape(feature_B, (b, c, h * w))
#         feature_B = feature_B.transpose(0, 2, 1)
#         feature_mul = feature_B @ feature_A
#         feature_mul= np.reshape(feature_mul, (b, h, w, h * w))
#         feature_mul = feature_mul.transpose(0, 1, 3, 2)
#         correlation_tensor = feature_mul.transpose(0, 2, 1, 3)
#         # correlation_tensor = np.ascontiguousarray(correlation_tensor)
#         return [correlation_tensor]


# cv.dnn_registerLayer('Correlation', CorrelationLayer)

# onnxmodel = "gmm.onnx"
# inp1 = np.load("inp1_gmm.npy")
# inp2 = np.load("inp2_gmm.npy")
# ref = np.load("out_gmm_theta.npy")

# net = cv.dnn.readNet(onnxmodel)

# net.setInput(inp1, "input.1")
# net.setInput(inp2, "input.18")
# out = net.forward()

# ref = np.load('out_gmm_theta.npy')


# # print(max(abs(out - ref)))

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
        print("X", grid_X.shape)
        print("Y", grid_Y.shape)
        warped_grid = apply_transformation(theta, np.concatenate((grid_X, grid_Y), axis=3), N, P_X, P_Y)
        print("warped_grid", warped_grid.shape)
        return warped_grid

if __name__ == "__main__":
    # X = np.random.uniform(-1, 1, size=[25, 1])
    # Y = np.random.uniform(-1, 1, size=[25, 1])
    # Li = compute_L_inverse(X, Y)
    theta = np.random.uniform(-1, 1, size=(1, 50))
    grid = postprocess(theta)