from tqdm import tqdm
import numpy as np
from models.inferrer import Inferrer


class RBDMInferrer(Inferrer):

    def __init__(self, hparams, model, data_manager, sess):
        super().__init__(hparams, model, data_manager, sess)

    def _infer(self, **kwargs):
        score_op = self.model.inference['score']
        pooled_contexts_op = self.model.inference['pooled_contexts']
        contexts = kwargs['contexts']
        responses = kwargs['responses']

        inference_typ = kwargs['inference_typ']

        data_gen = self.data_manager.make_instance(contexts=contexts,
                                                   responses=responses)

        if inference_typ == 'score':
            op = score_op
        elif inference_typ == 'embedding':
            op = pooled_contexts_op
        else:
            raise ValueError

        results = []
        for data in tqdm(data_gen):
            result = self.sess.run(op, feed_dict=data)
            results.extend(result)
        return np.array(results)


if __name__ == '__main__':
    import tensorflow as tf
    from models.rbdm.model import RetrievalBasedDialogModel
    from models.rbdm.data_manager import RBDMDataManager
    from ingredients import create_embedding
    from hparams import parse_hparams, save_hparams

    hparams = parse_hparams()
    save_hparams(hparams)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    data_manager = RBDMDataManager(hparams=hparams)

    embedding = create_embedding(pretrained_path=None,
                                 name='embedding',
                                 vocab_size=data_manager.vocab_size,
                                 embedding_size=hparams.embedding_size)

    model = RetrievalBasedDialogModel(hparams=hparams, sess=sess, embedding=embedding)
    inferrer = RBDMInferrer(hparams=hparams, model=model, data_manager=data_manager, sess=sess)

    contexts = ['뭐 하고 있어 ?', '뭐 전부다 그런 거지', '니가 너무 보고 싶다']
    responses = ['그냥 아무 것도 안 하고 있어', '나는 천재 다', '회사 에서 또 혼 났어']

    inference = inferrer.infer(contexts=contexts, responses=responses, inference_typ='score')
    print(inference[:len(contexts)])
