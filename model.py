import tensorflow as tf


class BertClassifier(tf.keras.Model):
    def __init__(self, bert, num_labels):
        super().__init__()
        self.bert = bert
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_labels, activation="softmax")

    def call(self, inputs, training=False):
        outputs = self.bert(inputs, training=training)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output, training=training)
        return self.classifier(x)
    