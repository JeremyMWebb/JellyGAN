import os
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt


def to_grid( real, fake, row):
    H = real.shape[1]
    W = real.shape[2]
    col = real.shape[0] // row
    spacer = 5
    
    if len( real.shape) == 3:
        channels = 1
    else:
        channels = real.shape[-1]    
    grid = np.ones(( spacer+2*col*H, row*W +row-1, channels))
    for j in range( col):
        k = col + j
        for i in range( row):
            grid[ j*H:(j+1)*H, i+i*W:i+(i+1)*W] = real[i+j*row]
            grid[ spacer+k*H: spacer+(k+1)*H, i+i*W:i+(i+1)*W] = fake[i+j*row]

    if grid.shape[-1] == 1:
        grid = np.concatenate(( grid, grid, grid), axis=-1)
    grid = np.clip( grid, 0, 1)
    return grid

def plot_grid( grid, path, epoch):
    Width = 15
    Height = int((Width / grid.shape[1]) * grid.shape[0])
    plt.figure( figsize=( Width, Height))
    plt.xticks([])
    plt.yticks([])
    if grid.shape[-1] == 1:
        plt.imshow( grid[:,:,0])
    else:
        plt.imshow( grid)
    plt.savefig('{}/Epoch_{}.png'.format( path, epoch))
    plt.close()

def profile_data( data_location):
    data = os.listdir( data_location)
    total = 0
    test = []
    for d in data:
        f = h5py.File( os.path.join( data_location, d))
        num = len( f['imgs_data'])
        total += num
        test.append([ d, np.arange( num)])
        f.close()
    return test
    
def plot_loss( crt_real, crt_fake, gen_loss, path):
    plt.figure(figsize=(8,8))
    plt.subplot(311)
    plt.xticks([]); plt.gca().set_title('Critic Real Loss')
    plt.plot( crt_real)

    plt.subplot(312)
    plt.xticks([]); plt.gca().set_title('Critic Fake Loss')
    plt.plot( crt_fake)
    
    plt.subplot(313)
    plt.gca().set_title('Generator Loss')
    plt.plot( gen_loss)
    plt.savefig('{}/Fig_Losses.png'.format( path))
    plt.close()    

def plot_accuracy( crt_acc, gen_acc, path):
    plt.figure()
    plt.subplot(211)
    plt.gca().set_title('Critic Accuracy')
    plt.plot( crt_acc)
    plt.xticks([]) ; plt.ylim( -0.05, 1.05)
    plt.subplot(212)
    plt.gca().set_title('Generator Accuracy')
    plt.plot( gen_acc)
    plt.xticks([]) ; plt.ylim( -0.05, 1.05)
    plt.savefig('{}/Fig_Accuracy.png'.format( path))
    plt.close()    
    
def SSIM( x, y):
    x = np.swapaxes( x, 0, -1)
    y = np.swapaxes( y, 0, -1)
    SSIM = []
    k1, k2, L = 0.01, 0.03, np.max( x)
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    c3 = c2 / 2
    for ch in range( x.shape[0]):
        muX = np.mean( x[ch], axis=(0,1))
        muY = np.mean( y[ch], axis=(0,1))
        sigX = np.sqrt( np.mean( np.square( x[ch] - muX), axis=(0,1)))
        sigY = np.sqrt( np.mean( np.square( y[ch] - muY), axis=(0,1)))
        l = (2* muX* muY+ c1) / (muX**2 + muY**2 + c1)
        c = (2* sigX* sigY+ c2) / (sigX**2 + sigY**2 + c2)
        s = np.mean(( x-muX)*( y-muY)+c3, axis=(0,1)) / ( sigX * sigY + c3)
        SSIM.append( np.mean( l* c* s))
    return np.mean( SSIM)

def avgpool( x):
    y = np.empty(( x.shape[0], x.shape[1]//2, x.shape[2]//2, x.shape[-1]))
    for i in range( 0, x.shape[1], 2):
        for j in range( 0, x.shape[2], 2):
            y[:,i//2,j//2,:] = np.mean( x[:, i:(i+1)*2, j:(j+1)*2, :], axis=(1,2))
    return y

def MS_SSIM( x, y):
    x = np.swapaxes( x, 0, -1)
    y = np.swapaxes( y, 0, -1)
    SSIM = 1
    while x.shape[1] > 4:    
        muX = np.mean( x, axis=(0,1,2))
        muY = np.mean( y, axis=(0,1,2))
        sigX = np.sqrt( np.mean( np.square( x - muX), axis=(0,1,2)))
        sigY = np.sqrt( np.mean( np.square( y - muY), axis=(0,1,2)))
        c = (2* sigX* sigY) / (sigX**2 + sigY**2 + 1e-6)
        s = np.mean(( x-muX)*( y-muY), axis=(0,1,2)) / ( sigX * sigY + 1e-6)
        SSIM *= c * s
        x = avgpool( x)
        y = avgpool( y)
    l = (2* muX* muY) / (muX**2 + muY**2 + 1e-6)

    SSIM = l * SSIM
    return np.mean( SSIM)

def to_one_hot( label, classes=10):
    size = label.shape
    new_label = np.zeros(( size[0], classes), dtype='float32')
    for i in range( size[0]):
        new_label[ i, label[i]] = 1
        
    return new_label

def sample( x_real, y_real, num, Block, num_class=10):
    size = 4* 2**Block
    channels = x_real.shape[-1]
    classes = np.arange( num * num_class)
    classes = classes // num
    latent = np.random.randn( num * num_class, 512).astype('float32')
    
    out_imgs = np.empty(( num * num_class, size, size, channels))
    select = np.empty( num * num_class, dtype='int')
    for i in range( num * num_class):
        loc = np.where( y_real == classes[i])[0]
        select[i] = np.random.choice( loc, 1)
    hold = x_real[ select]
    classes = to_one_hot( classes)

    for i in range( num * num_class):
        out_imgs[i] = np.reshape( cv2.resize( hold[i], (size, size)), (size, size, channels))
    out_imgs = out_imgs / 127.5 - 1.0
    return out_imgs, latent, classes    
    
def latent_generator( Batch_size):
    while True:
        latent = np.random.randn( Batch_size, 512).astype('float32')
        classes = np.random.randint( 0, 9, Batch_size)
        classes = to_one_hot( classes)
        yield latent, classes

def data_generator( x_data, y_data, Batch_size, Block):
    input_size = 4* 2**Block
    channels = x_data.shape[-1]
    while True:
        selection = np.random.choice( x_data.shape[0], Batch_size, replace=False)
        x_hold = np.zeros(( Batch_size, input_size, input_size, channels))
        y_hold = np.zeros(( Batch_size))
        
        for i in range( Batch_size):
            x_hold[i] = np.reshape( cv2.resize( x_data[ selection[i]], (input_size, input_size)), (input_size, input_size, channels))
            y_hold[i] = y_data[ selection[i]]

        out_labels = to_one_hot( y_hold.astype( int))
        out_imgs = x_hold / 127.5 - 1.0

        yield out_imgs, out_labels

def precheck( x_train, y_train):
    num_classes = len( np.unique( y_train))
    y_train = np.squeeze( y_train)
    size = x_train.shape[1]
    Channels = x_train.shape[-1]
    steps = 4
    
    sample = []
    grid = np.empty(( size, num_classes * size, Channels))
    for i in range( num_classes):
        grid[:, i*size:(i+1)*size] = x_train[ y_train == i][0]
    grid /= np.max( grid)

    for i in range( steps):
        sample.append( cv2.resize( grid, (grid.shape[1]//2**i, grid.shape[0]//2**i)))

    plt.figure()
    for i in range( steps):
        plt.subplot( steps, 1, i+1)
        plt.imshow( sample[i])
    plt.savefig('Classification_check.png')
    plt.close()
    
    ssim = []
    init = 4
    for j in range( steps):
        temp = []
        x1 = np.empty(( 250, init, init, Channels))
        x2 = np.empty(( 250, init, init, Channels))
        for i in range( num_classes):
            print( j, i)
            hold = x_train[ y_train == i]
            if Channels == 1:
                for k in range( 250):
                    x1[k,:,:,0] = cv2.resize( hold[k], (init, init))
                    x2[k,:,:,0] = cv2.resize( hold[250+k], (init, init))
            else:
                for k in range( 250):
                    x1[k] = cv2.resize( hold[k], (init, init))
                    x2[k] = cv2.resize( hold[250+k], (init, init))                
            
            temp.append( SSIM( x1, x2))
        init *= 2
        ssim.append( np.mean( temp))
        
    o = MS_SSIM( x1, x2)
    print( o)
    
    plt.figure()
    plt.plot( ssim)
    plt.savefig('SSIM_check.png')
    plt.close()
    
