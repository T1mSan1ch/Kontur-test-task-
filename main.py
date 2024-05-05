from solution_kontur import (
    Config,
    ImageProcessor,
    LabelManager,
    Model,
    Train,
    Predict
)
from datasets import (
    Dataset,
    load_dataset
)
import pandas as pd


PATH_TO_DATA = "Данные"
dataset = load_dataset("imagefolder", data_dir=PATH_TO_DATA)



def main(dataset):
    config = Config()
    image_processor = ImageProcessor(config=config, dataset=dataset)
    proc = image_processor.get_img_processor
    label_manager = LabelManager(dataset=dataset)
    model = Model(config=config, labels=label_manager)
    train = Train(config=config, preprocessor=image_processor, model=model)
    train.train()
    train.evaluate()
    train.push_to_hub()

    sample_submission = pd.read_csv('sample_submission.csv')
    preds = Predict(config=config, model=model)

    PATH = ''
    test = pd.read_csv('test.csv')
    mask = ~test['id'].str.endswith(('.png', '.jpg', '.jpeg', '.webp'))

    test.loc[mask, 'id'] += '.png'
    test['id'] = PATH + test['id']
    submit = preds.predict(test)
    submit['id'] = sample_submission['id']
    submit.to_csv('submit_w_classes.csv', index=False)


if __name__ == '__main__':
    main(dataset)
