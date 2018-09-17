from models.rbdm.trainer import RBDMTrainer

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

    embedding = create_embedding(pretrained_path=hparams.pretrained_path,
                                 name='embedding',
                                 vocab_size=data_manager.vocab_size,
                                 embedding_size=hparams.embedding_size)

    model = RetrievalBasedDialogModel(hparams=hparams, sess=sess, embedding=embedding)
    trainer = RBDMTrainer(hparams=hparams, model=model, data_manager=data_manager, sess=sess)

    trainer.train()
