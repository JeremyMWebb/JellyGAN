import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras.backend as K
from keras.models import Model
from keras.constraints import max_norm
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten, AveragePooling2D, UpSampling2D, Add, Concatenate, Lambda, LeakyReLU
channels = [512, 512, 512, 512, 256, 128, 64, 32, 16, 3, 3]

def PixNorm( x):
    return x / K.sqrt( K.mean( K.square( x), axis=-1, keepdims=True) + K.epsilon())

def MiniBatchSTD( x):
    s = K.shape( x)
    y = K.reshape( x, [ 4, -1, s[1], s[2], s[3]])
    y -= K.mean( y, axis=0, keepdims=True)
    y = K.mean( K.square(y), axis=0)
    y = K.sqrt( y + K.epsilon())
    y = K.mean( y, axis=[1,2,3], keepdims=True)
    y = K.tile( y, [ 4, s[1], s[2], 1])
    return K.concatenate(( x, y), axis=-1)

def toRGB( x, channel, block):
    x = Conv2D( channel, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', activation='linear', name='toRGB'+str(block))(x)
    return x

def up_block( y, feature, block, con):
    x = UpSampling2D(( 2, 2), name='b{}l1'.format( block))(y)
    x = Conv2D( feature, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b{}l2'.format( block))(x)
    x = LeakyReLU( alpha=0.2, name='b{}l3'.format( block))(x)
    x = Lambda( PixNorm, name='b{}l4'.format( block))(x)
    x = Conv2D( feature, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b{}l5'.format( block))(x)
    x = LeakyReLU( alpha=0.2, name='b{}l6'.format( block))(x)
    x = Lambda( PixNorm, name='b{}l7'.format( block))(x)
    return x, y
    
def Generator( Block, Transition, Channels):
    Alpha = K.variable( 0.0)
    def mult_alpha(x):
        return x * Alpha
        
    def mult_beta(x):
        return x * ( 1.00- Alpha)
    
    if Transition:
        con = 4.
    else:
        con = 100.
    
    latent_in = Input(( 512,), name='latent_input')

    x = Dense( 512* 4* 4, kernel_constraint=max_norm(con), name='latent')(latent_in)
    x = Reshape(( 4, 4, 512), name='reshape')(x)
    x = LeakyReLU( alpha=0.2, name='b0l1')(x)
    x = Lambda( PixNorm, name='b0l2')(x)
    x = Conv2D( 512, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l3')(x)
    x = LeakyReLU( alpha=0.2, name='b0l4')(x)
    x = Lambda( PixNorm, name='b0l5')(x)
    x = Conv2D( 512, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l6')(x)
    x = LeakyReLU( alpha=0.2, name='b0l7')(x)
    x = Lambda( PixNorm, name='b0l8')(x)
    
    for block in range( Block):
        x, y = up_block( x, channels[ block], block+1, con)
        
    if Transition:
        alpha = toRGB( x, Channels, Block)
        alpha = Lambda( mult_alpha, name='alpha')(alpha)
        
        beta = toRGB( y, Channels, Block-1)
        beta = UpSampling2D(( 2, 2), name='up_beta')(beta)
        beta = Lambda( mult_beta, name='beta')( beta)
        outputs = Add()([ alpha, beta])
    else:
        outputs = toRGB( x, Channels, Block)

    model = Model( latent_in, outputs)

    return model, Alpha

def fromRGB( x, feature, string):
    x = Conv2D( feature, (1, 1), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', activation='linear', name='fromRGB'+string)(x)
    x = LeakyReLU( alpha=0.2, name='actFromRGB'+string)(x)
    return x

def down_block( x, feature, block, con):
    x = Conv2D( feature, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name= 'b{}l1'.format( block))(x)
    x = LeakyReLU( alpha=0.2, name= 'b{}l2'.format( block))(x)
    x = Conv2D( feature, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name= 'b{}l3'.format( block))(x)
    x = LeakyReLU( alpha=0.2, name= 'b{}l4'.format( block))(x)
    x = AveragePooling2D(( 2, 2), name= 'b{}l5'.format( block))(x)
    return x

def Critic( Block, Transition, Channels):
    Alpha = K.variable( 0.0)
    def mult_alpha(x):
        return x * Alpha
        
    def mult_beta(x):
        return x * ( 1.00- Alpha)    

    if Transition:
        con = 4.
    else:
        con = 100.

    inputs = Input(( 4* 2**Block, 4* 2**Block, Channels), name='Gen_input')
    
    if Transition:
        alpha = fromRGB( inputs, channels[ Block+1], 'alpha')
        alpha = down_block( alpha, channels[ Block], Block, 100.)
        alpha = Lambda( mult_alpha, name='alpha')(alpha)
        
        beta = AveragePooling2D(( 2, 2), name='beta_down')(inputs)
        beta = fromRGB( beta, channels[ Block], 'beta')
        beta = Lambda( mult_beta, name='beta')(beta)
        x = Add()([ alpha, beta])
        Block -= 1
    else:
        x = fromRGB( inputs, channels[ Block+1], 'alpha')

    for block in range( Block, 0, -1):
        x = down_block( x, channels[ block], block, con)
        
    x = Lambda( MiniBatchSTD, name='MiniBatchSTD')(x)
    x = Conv2D( 512, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l1')(x)
    x = LeakyReLU( alpha=0.2, name='b0l2')(x)    
    x = Conv2D( 512, (3, 3), padding='same', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l3')(x)
    x = LeakyReLU( alpha=0.2, name='b0l4')(x)
    x = Flatten( name='flatten')(x)
    x = Dense( 512, kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l5')(x)
    x = LeakyReLU( alpha=0.2, name='b0l6')(x)
    outputs = Dense( 1, activation='linear', kernel_initializer='he_normal', bias_initializer='zeros', kernel_constraint=max_norm(con), name='b0l7')(x)
    
    model = Model( inputs, outputs)

    return model, Alpha

if __name__ == '__main__':
#     True False
    for i in range( 1, 3):
        for phase in [False, True]:
            model, alpha = Critic( i, phase, 1)
#            model.load_weights('test.h5', by_name=True)
#            model.save_weights('test.h5')
#    model, alpha = Generator( 2, True, 3)
    
            model.summary()
    
    