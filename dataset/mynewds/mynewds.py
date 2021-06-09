"""mynewds dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import re
import numpy as np

# TODO(mynewds): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(mynewds): BibTeX citation
_CITATION = """
Saeed Ahmadian
"""


class Mynewds(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mynewds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(mynewds): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'Dosimetric':tfds.features.Tensor(shape=(4,),dtype=tf.float32),
            'Non-Dosimetric': tfds.features.Tensor(shape=(30,),dtype=tf.float32),
            'Sparse_features':tfds.features.Tensor(shape=(8,),dtype=tf.float32),
            'Initial_seq':tfds.features.Tensor(shape=(2,1),dtype=tf.float32),
            'Target_seq':tfds.features.Tensor(shape=(4,1),dtype=tf.float32),
            # 'image': tfds.features.Image(shape=(None, None, 3)),
            # 'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,
        # ('Dosimetric', 'Non-Dosimetric',
        #                  'Sparse_features','Initial_seq','Target_seq'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(mynewds): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    path = Path('C:\\Users\\saeed.ahmadian\\PycharmProjects\\TFExplore\\datapath')

    # TODO(mynewds): Returns the Dict[split names, Iterator[Key, Example]]
    return [tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "path": path
            })]
    #     {
    #     # 'train': self._generate_examples(path / 'train_imgs'),
    #     'train_path': self._generate_examples(path),
    # }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(mynewds): Yields (key, example) tuples from the dataset
    i=0
    for file in path.glob('*.csv'):
        with tf.io.gfile.GFile(file) as f:
            for line in f:
                line= re.sub('\n','',line)
                line= re.sub(', ,','',line)
                line= re.sub(',[0-9]+\-[a-zA-Z]{3}\-[0-9]+,','',line)
                line = re.sub(',[0-9]+\.[a-zA-Z]+\.[0-9]+,', '', line)
                line = re.sub('[0-9]+[.][0-9]+[.][0-9]+','',line)
                if re.match('[a-zA-Z]',line) is None:
                    rawline=line.split(',')
                    tmp = list(map(lambda x: 0 if re.match('[0-9]+[.]*[0-9]*',x) is None else float(x), rawline))
                    yield i, {
                        'Dosimetric': np.reshape(np.array(tmp[0:4],dtype=np.float32),(4,)),
                        'Non-Dosimetric': np.reshape(np.array(tmp[4:34],dtype=np.float32),(30,)),
                        'Sparse_features': np.reshape(np.array(tmp[34:42],dtype=np.float32),(8,)),
                        'Initial_seq': np.reshape(np.array(tmp[42:44],dtype=np.float32),(2,1)),
                        'Target_seq': np.reshape(np.array(tmp[44:48],dtype=np.float32),(4,1))
                    }
                    i+=1




    # for f in path.glob('*.jpeg'):
    #   yield 'key', {
    #       'image': f,
    #       'label': 'yes',
    #   }
