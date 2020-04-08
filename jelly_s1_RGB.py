import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import h5py
import numpy as np
import skimage.transform
import keras.backend as K
from functools import partial
from keras.models import Model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from Networks_paper import Generator, Critic
from Support import plot_grid, plot_loss, to_grid

def gradient_penalty_loss( y_true, y_pred, averaged_samples):
    gamma = 500.
    grad = K.gradients( y_pred, averaged_samples)[0]
    norm = K.sqrt( K.sum( K.square( grad)))
    loss = K.square( norm - gamma) / K.square( gamma)
    return loss

def wass_loss( y_true, y_pred):
    return K.mean( y_true * y_pred)

def RandomWeightedAverage( x):
    alpha = K.random_uniform(( Batch_size, 1, 1, 1))
    return alpha* x[0] + (1- alpha)* x[1]
        
def build_models( Block, Transition, Learning_rate, Channels):
    Gen, AlphaG = Generator( Block, Transition, Channels)
    Crt, AlphaC = Critic( Block, Transition, Channels)

    Crt.trainable = False
    Gen.trainable = True
    latent_in0 = Input(( 512, ))
    fake_img0 = Gen( latent_in0)
    source_out0 = Crt( fake_img0)

    Gen_train = Model( latent_in0, source_out0)
    Gen_train.compile( RMSprop( lr=Learning_rate), loss= wass_loss)

    Crt.trainable = True
    Gen.trainable = False
    real_in1 = Input(( 4*2**Block, 4*2**Block, Channels))
    latent_in1 = Input(( 512, ))
    fake_in1 = Gen( latent_in1)
    mixed_in1 = Lambda( RandomWeightedAverage)([ real_in1, fake_in1])
    real_outC  = Crt( real_in1)
    fake_outC  = Crt( fake_in1)
    mixed_outC = Crt( mixed_in1)

    partial_gp_loss = partial( gradient_penalty_loss, averaged_samples=mixed_in1)
    partial_gp_loss.__name__ = 'gradient_penalty'

    Crt_train = Model( [real_in1, latent_in1], [real_outC, fake_outC, mixed_outC])
    Crt_train.compile( RMSprop( lr=Learning_rate), loss= [wass_loss, wass_loss, partial_gp_loss], loss_weights=[1, 1, 10])

    Gen.compile( RMSprop( lr=Learning_rate), loss='mse')
    Crt.compile( RMSprop( lr=Learning_rate), loss='mse')
    
    return Gen_train, Crt_train, Gen, Crt, AlphaG, AlphaC


def critic_generator( Batch_size, Height, Width, Channels):
    f = h5py.File('RGBjelliesAug5.h5','r')
    data = f['data']
    while True:
        imgs = np.empty(( Batch_size, Width, Height, Channels), dtype='float32')
        for i in range( Batch_size):
            imgs[i] = skimage.transform.resize( data[ np.random.randint( 0, len( data))], (Width, Height))
            
        yield imgs

def train( Continue, init_block, final_block, dest_folder,
    Epochs = 40,
    Channels = 3,
    interval = 5,
    eval_steps = 10,
    Steps_per_epoch = 512,
    Learning_rate = 0.0005,
    Early_stopping_patience = 8):

    Phases = [False, True]
    Tcrt_real, Tcrt_fake, Tgen_loss = [], [], []
    for i in range( init_block, final_block):
        K.clear_session()
        Block = (i+1)//2
        Transition = Phases[ i % 2]
        Height = 4*2**Block
        Width = 4*2**Block
        dest_path = '{}/{}_{}_{}_{}x{}'.format( dest_folder, i, Block, Transition, Height, Width)
        prev_path = '{}/{}_{}_{}_{}x{}'.format( dest_folder, i-1, (i+0)//2, not Transition, 4*2**((i+0)//2), 4*2**((i+0)//2))
        init_alpha = 0.0
        init_epoch = 0
            
        if Continue:
            state = np.load('{}/state.npy'.format( dest_path), allow_pickle=True)
            init_alpha = state[1]
            init_epoch = int( state[0])
            Tcrt_real = state[2]
            Tcrt_fake = state[3]
            Tgen_loss = state[4]
#            crt_real = state[5]
#            crt_fake = state[6]
#            gen_loss = state[7]      
            crt_real, crt_fake, gen_loss = [], [], []
        elif not Continue and Block > 0:
            state = np.load('{}/state.npy'.format( prev_path), allow_pickle=True)
            Tcrt_real = state[2]
            Tcrt_fake = state[3]
            Tgen_loss = state[4]
            crt_real, crt_fake, gen_loss = [], [], []

        if not os.path.isdir( dest_path):
            os.mkdir( dest_path)
        
        Gen_train, Crt_train, Gen, Crt, AlphaG, AlphaC = build_models( Block, Transition, Learning_rate, Channels)
        if Block > 0 and not Continue:
            Gen.load_weights('{}/weight_generator.h5'.format( prev_path), by_name=True, skip_mismatch=True)
            Crt.load_weights('{}/weight_critic.h5'.format( prev_path), by_name=True, skip_mismatch=True)
        if Continue:
            Gen.load_weights('{}/weight_generator.h5'.format( dest_path))
            Crt.load_weights('{}/weight_critic.h5'.format( dest_path))
            Continue = False

        Alpha = init_alpha
        train_critic = critic_generator( Batch_size, Height, Width, Channels) 
        Y = np.ones(( Batch_size, 1)).astype('float32')
        for epoch in range( init_epoch, Epochs+1):
###########################  TRANSITION AND GRAPH  ############################
            if epoch % interval == 0:
                np.save('{}/state.npy'.format( dest_path), [interval * (epoch//interval), Alpha, Tcrt_real, Tcrt_fake, Tgen_loss, crt_real, crt_fake, gen_loss])
                Gen.save_weights('{}/weight_generator.h5'.format( dest_path))
                Crt.save_weights('{}/weight_critic.h5'.format( dest_path))
                
                z = np.random.randn( Batch_size, 512).astype('float32')
                x_real = next( train_critic)
                x_fake = Gen.predict( z)
                
                x_real = (x_real + 1.0) / 2.0
                x_fake = (x_fake + 1.0) / 2.0
                grid = to_grid( x_real, x_fake, 16)
                plot_grid( grid, dest_path, epoch)
                plot_loss( crt_real, crt_fake, gen_loss, dest_path)
                plot_loss( Tcrt_real, Tcrt_fake, Tgen_loss, dest_folder)

            print('Epoch {} of {}'.format( epoch, Epochs))
            cr, cf, gl = [], [], []
            for itr in range( Steps_per_epoch):
###################################  STEP #####################################
                if Transition:
                    Alpha += 1.0 / ((Epochs+1.0) * Steps_per_epoch)
                    K.set_value( AlphaC, Alpha)
                    K.set_value( AlphaG, Alpha)

################################  TRAIN CRITIC  ###############################
                imgs = next( train_critic)
                z = np.random.randn( Batch_size, 512).astype('float32')
                outC = Crt_train.train_on_batch( [imgs, z], [Y, -Y, Y])
                cr.append( outC[1])
                cf.append( outC[2])

#############################  TRAIN GENERATOR  ###############################
                z = np.random.randn( Batch_size, 512).astype('float32')
                outG = Gen_train.train_on_batch( z, Y)
                gl.append( outG)
            crt_real.append( np.mean( cr))
            crt_fake.append( np.mean( cf))
            gen_loss.append( np.mean( gl))
            Tcrt_real.append( crt_real[-1])
            Tcrt_fake.append( crt_fake[-1])
            Tgen_loss.append( gen_loss[-1])
            print('ALPHA {:5.4f}'.format( Alpha))
            print('CRITIC Rl {:3.2f}, Fl {:3.2f}, Ml {:3.2f}'.format( outC[1], outC[2], outC[3]))
            print('GENERATOR Gen loss {:3.2f}\n'.format( outG))


if __name__ == '__main__':
    Continue = True
    
    interval = 5
    Steps = 32
    Epochs = 400
    Batch_size = 32
    
    final_block = 7
    init_block = 6

    Channels = 3
    Learning_rate = 0.0001

    dest_folder = 'Results_RGB_jellies_s1'
    if not os.path.isdir( dest_folder):
        os.mkdir( dest_folder)

    train( Continue, init_block, final_block, dest_folder, Steps_per_epoch=Steps, Epochs=Epochs, Channels=Channels, Learning_rate=Learning_rate, interval=interval)
    
