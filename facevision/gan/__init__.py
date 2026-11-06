"""生成对抗网络 (GAN) 相关模块: StarGAN 人脸属性编辑 (发色 / 性别 / 年龄)。"""
from .stargan import (ATTRS, PRESETS, SINGLE_EDITS, Discriminator, Generator, StarGANEditor,
                      celeba_transform, load_discriminator, load_generator, make_target,
                      official_test_labels, use_instance_stats)

__all__ = ["ATTRS", "PRESETS", "SINGLE_EDITS", "Generator", "Discriminator", "StarGANEditor",
           "celeba_transform", "load_generator", "load_discriminator", "make_target",
           "official_test_labels", "use_instance_stats"]
