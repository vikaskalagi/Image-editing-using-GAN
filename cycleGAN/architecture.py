import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()

def instance_norm(inp):

    with tf.variable_scope("instance_norm"):
        eps = 1e-5
        mean, var = tf.nn.moments(inp, [1,2], keep_dims=True)
        scale = tf.get_variable('scale', [inp.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0,stddev=0.02))
        offset = tf.get_variable('offset', [inp.get_shape()[-1]], initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(inp-mean, tf.sqrt(var+eps)) + offset
        
        return out


def convdconv(input_conv, filters=64, kernel=7, stride=1, padding='VALID', name="conv", norm = True, stddev=0.02, relu = True, alpha = 0, convordconv=1):
    w_init = tf.truncated_normal_initializer(mean=0.0,stddev=0.02)
    b_init = tf.constant_initializer(0.0)

    with tf.variable_scope(name):
        if convordconv == 2:
            conv = tf.layers.conv2d_transpose(input_conv, filters, kernel, stride, padding, kernel_initializer=w_init, bias_initializer=b_init)
        else:
            conv = tf.layers.conv2d(input_conv, filters, kernel, stride, padding, kernel_initializer=w_init, bias_initializer=b_init)
        
        if norm == True:
            conv = instance_norm(conv)
        
        if relu == True:
            if alpha == 0:
                conv = tf.nn.relu(conv)
            else:
                conv = tf.nn.leaky_relu(conv, alpha=alpha)
        
        return conv


def resnet_block(input_res, name="resnet"):
    with tf.variable_scope(name):
        output_res = tf.pad(input_res, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        output_res = convdconv(output_res, filters=128, kernel=3, name="c1")
        output_res = tf.pad(output_res, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        output_res = convdconv(output_res, filters=128, kernel=3, name="c2", relu=False)
        
        return tf.nn.relu(output_res + input_res)    


def generator(input_gen, name='generator'):
    
    with tf.variable_scope(name):
        
        pad_input = tf.pad(input_gen, [[0,0],[3,3],[3,3],[0,0]], mode="REFLECT")
        
        d32 = convdconv(pad_input, filters=32, name="conv1")
        d64 = convdconv(d32, kernel=3, stride=2, padding="SAME", name="conv2")
        d128 = convdconv(d64, filters=128, kernel=3, stride=2, padding="SAME", name="c3")
        
        R128_1 = resnet_block(d128, name="r1")
        R128_2 = resnet_block(R128_1, name="r2")
        R128_3 = resnet_block(R128_2, name="r3")
        R128_4 = resnet_block(R128_3, name="r4")
        R128_5 = resnet_block(R128_4, name="r5")
        R128_6 = resnet_block(R128_5, name="r6")
        
        u64 = convdconv(R128_6, kernel=3, stride=2, padding="SAME", name="dc1", convordconv=2)
        u32 = convdconv(u64, filters=32, kernel=3, stride=2, padding="SAME", name="dc2", convordconv=2)
        
        u32_pad = tf.pad(u32, [[0,0],[3,3],[3,3],[0,0]], mode="REFLECT")
        
        c7s1_3 = convdconv(u32_pad, filters=3, name="c4", relu=False)
        
        output_gen = tf.nn.tanh(c7s1_3,"out_gen")
        
        return output_gen
    

def discriminator(input_disc, name="discriminator"):
    
    with tf.variable_scope(name):
        
        C64 = convdconv(input_disc, kernel=4, stride=2, padding="SAME", name="c1", norm=False, alpha=0.2)
        C128 = convdconv(C64, filters=128, kernel=4, stride=2, padding="SAME", name="c2", alpha=0.2)
        C256 = convdconv(C128, filters=256, kernel=4, stride=2, padding="SAME", name="c3", alpha=0.2)
        C512 = convdconv(C256, filters=512, kernel=4, stride=2, padding="SAME", name="c4", alpha=0.2)

        logits = convdconv(C512, filters=1, kernel=4, padding="SAME", name="disc_logits", norm=False, relu=False)

        return logits
