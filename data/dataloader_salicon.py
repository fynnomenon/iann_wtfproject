import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # filter out info and warning messages
from pycocotools.coco import COCO
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import scipy
import sys

data_dir = './SALICON'
data_type = str(sys.argv[1])

tfds_path = f'{data_dir}/tfds_salicon/{data_type}'
cap_file = f'{data_dir}/annotations/{data_type[:-4]}/captions_{data_type}.json'
fix_file = f'{data_dir}/annotations/{data_type[:-4]}/fixations_{data_type}.json'

# Initialize COCO API for instance annotations
salicon = COCO(fix_file)

# Load informations for all the images that occur in SALICON
img_ids = salicon.getImgIds();
imgs = salicon.loadImgs(img_ids)

# Load fixations for all the images that occur in SALICON
fix_ids = salicon.getAnnIds(img_ids)
fixs = salicon.loadAnns(fix_ids)

# Initialize COCO API for instance annotations
coco_caps = COCO(cap_file)

# Load MS COCO captions for all the images that occur in SALICON as well
cap_ids = coco_caps.getAnnIds(img_ids)
caps = coco_caps.loadAnns(cap_ids)

bert_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
bert_preprocess_model = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1')

def get_sal_map(fixs, image_id, height, width):
    fixations = [fix['fixations'] for fix in fixs if fix['image_id'] == image_id]
    merged_fixations = [item for sublist in fixations for item in sublist]

    sal_map = np.zeros((height, width))
    for y,x in merged_fixations:
        sal_map[y-1][x-1] = 1

    blurred_sal_map = scipy.ndimage.gaussian_filter(sal_map, 19)
    normalized_sal_map = (blurred_sal_map-np.min(blurred_sal_map))/(np.max(blurred_sal_map)-np.min(blurred_sal_map)) # normalize to between 0 and 1
    normalized_sal_map = np.expand_dims(normalized_sal_map, axis = -1)
    resized_sal_map = tf.image.resize(normalized_sal_map, [224, 224])

    return resized_sal_map

def get_cap(caps, image_id):
    captions = np.asarray([cap['caption'] for cap in caps if cap['image_id'] == image_id])
    rand_state = np.random.RandomState(61)
    cap = rand_state.choice(captions, 1)
    encoder_inputs = bert_preprocess_model(cap)
    encoder_outputs = bert_model(encoder_inputs)
    sen_embed = encoder_outputs['sequence_output'][:,0,:] # take the hidden states of the [CLS] token of the last layer

    return sen_embed

def get_img(img, data_dir, data_type):
    coco_img = tf.image.decode_jpeg(tf.io.read_file(f'{data_dir}/images/{data_type[:-4]}/{img["file_name"]}'), channels=3)
    resized_img = tf.image.resize(coco_img, [224, 224])
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(resized_img) # convert image from RGB to BGR and zero-centre the color channels

    return preprocessed_img

def data_generator(imgs, fixs, caps, data_dir, data_type):
    for img in imgs:
        saliency_map = get_sal_map(fixs, img['id'], img['height'], img['width'])
        sentence_embedding = get_cap(caps, img['id'])
        image = get_img(img, data_dir, data_type)

        yield image, sentence_embedding, saliency_map

ds = tf.data.Dataset.from_generator(
    lambda: data_generator(imgs, fixs, caps, data_dir, data_type),
    output_types= (tf.float32, tf.int32, tf.float32),
    output_shapes=((224, 224, 3), (1, 768), (224, 224, 1))
)

is_exist = os.path.exists(tfds_path)
if not is_exist:
    os.makedirs(tfds_path)

ds.save(tfds_path, compression='GZIP')
