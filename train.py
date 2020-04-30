import os
from model.model import ModelName
import tensorflow as tf
from data_generator.data_generator import DataGenerator
from model.training.trainer import Trainer


def main(project):
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_kwargs = dict(batchsize_tr=1,
                        num_class=15,
                        )

    data_provider = DataGenerator( prefix=base_input,  data_kwargs=data_kwargs,)
    data_provider.gen_train_data()

    model_kwargs = dict(model="transformer_gkv",
                            char_dimension=100,
                            num_edge=6,
                            )

    model = ModelName(data_provider.num_class,
                            model_kwargs
                            )

    opt_kwargs = dict(optimizer="radam",
                        learning_rate=0.005
                        )
    cost_kwargs = dict(cost_name="cross_entropy",
                        act_name="softmax",
                        is_weight=False,
                        class_weights=False,
                        )

    trainer = Trainer(model,
                        labels=data_provider.list_val_data[2],
                        gpu_device="0",
                        path_weight=path_weight,
                        opt_kwargs=opt_kwargs,
                        cost_kwargs=cost_kwargs
                        )
    trainer.train(data_provider,
                    epochs=1005,
                    num_text_feature=data_provider.num_text_feature,
                    batch_steps_per_epoch=256,
                    is_restore=True
                    )

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--project',
                        default='nttd')

    args = parser.parse_args()
    main(args.project)
