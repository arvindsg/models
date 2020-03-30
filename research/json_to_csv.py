import os
import sys
import glob
import pandas as pd
import json

print("**")


def json_to_csv():
    print("***********************************************")
    json_list = []
    labels = []

    project_name = sys.argv[1]
    project_path = sys.argv[2]
    num_steps = sys.argv[3]
    print(project_name)
    print(project_path)

    print(os.getcwd())
    for directory in ['train', 'test']:
        image_path = os.path.join('{0}/images/{1}'.format(project_path, directory))
        print(image_path)
        for json_file in glob.glob(image_path + '/*.json'):
            print(json_file)
            j_name = os.path.basename(json_file)
            with open(json_file) as json_file:
                data = json.load(json_file)
                for shape in data["shapes"]:
                    value = []
                    value.append(j_name)
                    value.append(data["imagePath"])
                    value.append(data["imageWidth"])
                    value.append(data["imageHeight"])
                    value.append(shape["label"])
                    xmin = min(shape["points"][0][0], shape["points"][1][0])
                    xmax = max(shape["points"][0][0], shape["points"][1][0])
                    ymin = min(shape["points"][0][1], shape["points"][1][1])
                    ymax = max(shape["points"][0][1], shape["points"][1][1])

                    value.append(xmin)
                    value.append(ymin)
                    value.append(xmax)
                    value.append(ymax)


                    #value.append(shape["points"][0][0])
                    #value.append(shape["points"][0][1])
                    #value.append(shape["points"][1][0])
                    #value.append(shape["points"][1][1])
                    labels.append(shape["label"])
                    json_list.append(value)

        column_name = ['json_name','filename', 'width', 'height', 'class','xmin', 'ymin', 'xmax', 'ymax']
        json_df = pd.DataFrame(json_list, columns=column_name)
        json_df.to_csv('{0}/data/{1}_{2}_labels.csv'.format(project_path, project_name, directory),index=None)
        print('Successfully converted json to csv in {}'.format(directory))


        json_list = []

    labels = set(labels)
    label_df = pd.DataFrame(labels, columns=["LABELS"])
    label_df.to_csv('{0}/training/labels.csv'.format(project_path),index=None)

    train_input_path = os.path.join(os.getcwd(), project_path, "data/train.record")

    label_map_path = os.path.join(os.getcwd(), project_path, "training/labels.pbtxt")

    test_input_path = os.path.join(os.getcwd(), project_path, "data/test.record")

    config_file_data = """# Faster R-CNN with Resnet-101 (v1) configuration for MSCOCO Dataset.
# Users should configure the fine_tune_checkpoint field in the train config as
# well as the label_map_path and input_path fields in the train_input_reader and
# eval_input_reader. Search for "PATH_TO_BE_CONFIGURED" to find the fields that
# should be configured.

model {
  faster_rcnn {
    num_classes: %d
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 5224
      }
    }
    feature_extractor {
      type: 'faster_rcnn_resnet101'
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        scales: [0.25, 0.5, 1.0, 2.0]
        aspect_ratios: [0.5, 1.0, 2.0]
        height_stride: 16
        width_stride: 16
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
  batch_size: 1
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0003
          schedule {
            step: 0
            learning_rate: .0003
          }
          schedule {
            step: 900000
            learning_rate: .00003
          }
          schedule {
            step: 1200000
            learning_rate: .000003
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "%s/object_detection/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt"
  from_detection_checkpoint: true
  # Note: The below line limits the training process to 200K steps, which we
  # empirically found to be sufficient enough to train the pets dataset. This
  # effectively bypasses the learning rate schedule (the learning rate will
  # never decay). Remove the below line to train indefinitely.
  num_steps: %d
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "%s"
  }
  label_map_path: "%s"
}

eval_config: {
  num_examples: 9
  # Note: The below line limits the evaluation process to 10 evaluations.
  # Remove the below line to evaluate indefinitely.
  max_evals: 100
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "%s"
  }
  label_map_path: "%s"
  shuffle: true
  num_readers: 1
  num_epochs: 1
}
""" %(len(labels), os.getcwd(), int(num_steps), str(train_input_path), str(label_map_path), str(test_input_path), str(label_map_path))

    with open('{0}/training/config-resnet-101.config'.format(project_path), 'w') as configfile:
        configfile.write(config_file_data)


json_to_csv()
