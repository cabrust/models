# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

import tensorflow as tf

from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import input_reader_pb2

parallel_reader = tf.contrib.slim.parallel_reader


def build(input_reader_config):
  """Builds a tensor dictionary based on the InputReader config.

  Args:
    input_reader_config: A input_reader_pb2.InputReader object.

  Returns:
    A tensor dict based on the input_reader_config.

  Raises:
    ValueError: On invalid input reader proto.
    ValueError: If no input paths are specified.
  """
  if not isinstance(input_reader_config, input_reader_pb2.InputReader):
    raise ValueError('input_reader_config not of type '
                     'input_reader_pb2.InputReader.')

  if input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader':
    config = input_reader_config.tf_record_input_reader
    if not config.input_path:
      raise ValueError('At least one input path must be specified in '
                       '`input_reader_config`.')
    _, string_tensor = parallel_reader.parallel_read(
        config.input_path[:],  # Convert `RepeatedScalarContainer` to list.
        reader_class=tf.TFRecordReader,
        num_epochs=(input_reader_config.num_epochs
                    if input_reader_config.num_epochs else None),
        num_readers=input_reader_config.num_readers,
        shuffle=input_reader_config.shuffle,
        dtypes=[tf.string, tf.string],
        capacity=input_reader_config.queue_capacity,
        min_after_dequeue=input_reader_config.min_after_dequeue)

    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        label_map_proto_file=label_map_proto_file)
    return decoder.decode(string_tensor)

  elif input_reader_config.WhichOneof('input_reader') == 'tf_record_input_reader_mux':
    mux_config = input_reader_config.tf_record_input_reader_mux
    
    # Enumerate TFRecordInputReaders
    with tf.name_scope('Files_Are_Read_Here'):
      string_tensors = []
      weights = []
      for index, config in enumerate(mux_config.tf_record_input_reader):
        weight = mux_config.input_weight[index]

        print('Multiplexed input reader %d (weight: %0.2f):' % (index, weight))
        for path in config.input_path:
          print('    %s' % path)

        for path in config.input_path[:]:
            print('    %s' % path)


        # Normal TFRecordInputReader stuff here
        _, string_tensor = parallel_reader.parallel_read(
            config.input_path[:],  # Convert `RepeatedScalarContainer` to list.
            reader_class=tf.TFRecordReader,
            num_epochs=(input_reader_config.num_epochs
                        if input_reader_config.num_epochs else None),
            num_readers=input_reader_config.num_readers,
            shuffle=input_reader_config.shuffle,
            dtypes=[tf.string, tf.string],
            capacity=input_reader_config.queue_capacity,
            min_after_dequeue=input_reader_config.min_after_dequeue)

        
        string_tensors += [string_tensor]
        weights += [weight]
      
    # Multiplex
    with tf.name_scope('Multiplexing_Happens_Here'):  
      current_slice = tf.multinomial(tf.log([weights]), 1)[0]
      # current_slice_prt = tf.Print(current_slice, [current_slice], 'Current Slice: ')
      out_string_tensor = tf.slice(string_tensors,current_slice, tf.constant([1], dtype=tf.int64))

    # Decode
    label_map_proto_file = None
    if input_reader_config.HasField('label_map_path'):
      label_map_proto_file = input_reader_config.label_map_path
    decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        label_map_proto_file=label_map_proto_file)
    return decoder.decode(out_string_tensor)

  raise ValueError('Unsupported input_reader_config.')
