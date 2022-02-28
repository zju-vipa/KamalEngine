dependencies = ['torch', 'kamal']

from kamal.vision.models.classification.cifar.wrn import wrn_40_2

if __name__=='__main__':
    import kamal, torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')

    args = parser.parse_args()
    model = wrn_40_2(num_classes=100)
    model.load_state_dict( torch.load(args.ckpt) )
    kamal.hub.save(
        model,
        save_path='exported/classification_amal_wrn_40_2',
        entry_name='wrn_40_2',
        spec_name=None,
        code_path=__file__,
        metadata=kamal.hub.meta.Metadata(
            name='wrn_40_2',
            dataset='CIFAR-10',
            task=kamal.hub.meta.TASK.CLASSIFICATION,
            url='https://github.com/zju-vipa/KamalEngine',
            input=kamal.hub.meta.ImageInput(
                size=32,
                range=[0, 1],
                space='rgb',
                normalize=dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            ),
            entry_args=dict(num_classes=10),
            other_metadata=dict(num_classes=10),
        )
    )
