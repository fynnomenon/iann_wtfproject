import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # filter out info and warning messages
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import scipy
import json

data_dir = './capgaze1'
tfds_path = f'{data_dir}/tfds_capgaze1'
cap_path = f'{data_dir}/transcribed_text'
fix_path = f'{data_dir}/gaze_converted_2'
img_path = f'{data_dir}/images'

img_ids = os.listdir(img_path)
img_ids = [img[:-4] for img in img_ids]

bert_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
bert_preprocess_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')

def get_sal_map(image_id, fix_path, height, width):
    fixations = []
    for i in range(1,6):
        fix_file = f'{fix_path}/{i}/gaze/{image_id}.npy'
        fix_dict = np.load(fix_file, allow_pickle=True, encoding='latin1').item()
        if fix_dict['dominant_eye'] == 0:
            fixations.append(fix_dict['left_eye'])
        else:
            fixations.append(fix_dict['right_eye'])
    merged_fixations = [item for sublist in fixations for item in sublist]
    
    sal_map = np.zeros((height, width))
    N1 = int(len(merged_fixations)/4)
    for j in range(N1-3):
        temp = merged_fixations[j*4+10]
        if temp and temp[0]>0 and temp[0]<1 and temp[1]>0 and temp[1]<1:
            x = int(temp[0]*width)
            y = int(temp[1]*height)
            sal_map[y,x] = 1
            
    sal_map = scipy.ndimage.gaussian_filter(sal_map, 19)
    sal_map = (sal_map-np.min(sal_map))/(np.max(sal_map)-np.min(sal_map)) # normalize to between 0 and 1
    sal_map = np.expand_dims(sal_map, axis = -1)
    sal_map = tf.image.resize(sal_map, [224, 224])
 
    return sal_map

def get_cap(image_id, cap_path, rng):
    i = rng.integers(low=1, high=6, size=6)[0]
    cap_file = f'{cap_path}/{i}/{image_id}.json'
    with open(cap_file) as f:
        cap = json.load(f)['text']
    encoder_inputs = bert_preprocess_model([cap])
    encoder_outputs = bert_model(encoder_inputs)
    sen_embed = encoder_outputs['sequence_output'][:,0,:] # take the hidden states of the [CLS] token of the last layer

    return sen_embed

def get_img(image_id, img_path):
    capgaze_img = tf.image.decode_jpeg(tf.io.read_file(f'{img_path}/{image_id}.jpg'), channels=3)
    orig_h, orig_w = capgaze_img.shape[:2]
    resized_img = tf.image.resize(capgaze_img, [224, 224])
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(resized_img) # convert image from RGB to BGR and zero-centre the color channels

    return preprocessed_img, orig_h, orig_w

rng = np.random.default_rng(61)
def data_generator(img_ids, img_path, fix_path, cap_path, rng):
    for idx in img_ids:
        image, h, w = get_img(idx, img_path)
        saliency_map = get_sal_map(idx, fix_path, h, w)
        caption = get_cap(idx, cap_path, rng)

        yield image, caption, saliency_map

ds = tf.data.Dataset.from_generator(
    lambda: data_generator(img_ids, img_path, fix_path, cap_path, rng),
    output_types= (tf.float32, tf.int32, tf.float32),
    output_shapes=((224, 224, 3), (1, 768), (224, 224, 1))
)

is_exist = os.path.exists(tfds_path)
if not is_exist:
    os.makedirs(tfds_path)

ds.save(tfds_path, compression='GZIP')