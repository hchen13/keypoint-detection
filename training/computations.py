import tensorflow as tf


def preprocess_input(x):
    return tf.cast(x, tf.float32) / 127.5 - 1.


@tf.function
def balanced_crossentropy(y_true, y_pred):
    ones = tf.reduce_sum(y_true)
    size = tf.size(y_true, out_type=tf.float32)
    beta = 1 - ones / size
    loss = -(beta * y_true * tf.math.log(y_pred + 1e-8) + (1 - beta) * (1 - y_true) * tf.math.log(1 - y_pred + 1e-8))
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def dice_loss(y_true, y_pred):
    _epsilon = 1e-5
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + _epsilon
    loss = 1. - 2 * intersection / union
    return loss


@tf.function
def l1_loss(y_true, y_pred, mask):
    l1_dist = tf.abs(y_true - y_pred)
    l1_loss_all = l1_dist[..., ::2] + l1_dist[..., 1::2]
    l1_loss = l1_loss_all * mask
    n = tf.reduce_sum(mask)
    loss = tf.reduce_sum(l1_loss) / n
    return loss


@tf.function
def compute_loss(y_true, y_pred):
    h_true, o_true = y_true
    h_pred, o_pred = y_pred
    h_loss = dice_loss(h_true, h_pred)
    o_loss = l1_loss(o_true, o_pred, h_true)
    return h_loss, o_loss


@tf.function
def train_on_batch(model, batch_data, optimizer):
    batch_images, h_true, o_true = batch_data   # h = heatmap, o = offsets
    x = preprocess_input(batch_images)
    with tf.GradientTape() as tape:
        h_pred, o_pred, identity = model(x)
        h_loss, o_loss = compute_loss(y_true=(h_true, o_true), y_pred=(h_pred, o_pred))
        total_loss = h_loss + o_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    report = dict(
        total_loss=total_loss,
        heatmap_loss=h_loss,
        offset_loss=o_loss
    )
    return report


def examine(model, dataset, limit=None):
    total_loss = tf.metrics.Mean(name='total_loss')
    heatmap_loss = tf.metrics.Mean(name='heatmap_loss')
    offset_loss = tf.metrics.Mean(name='offset_loss')
    accuracy = tf.metrics.BinaryAccuracy(name='acc')
    precision = tf.metrics.Precision(name='precision')
    recall = tf.metrics.Recall(name='recall')
    for i, (images, h_true, o_true) in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        images = preprocess_input(images)
        h_pred, o_pred, identity = model.predict(images)
        h_loss, o_loss = compute_loss(y_true=(h_true, o_true), y_pred=(h_pred, o_pred))
        t_loss = h_loss + o_loss
        total_loss.update_state(t_loss)
        heatmap_loss.update_state(h_loss)
        offset_loss.update_state(o_loss)
        accuracy.update_state(h_true, h_pred)
        precision.update_state(h_true, h_pred)
        recall.update_state(h_true, h_pred)

    return dict(
        total_loss=total_loss.result().numpy(),
        heatmap_loss=heatmap_loss.result().numpy(),
        offset_loss=offset_loss.result().numpy(),
        accuracy=accuracy.result().numpy(),
        precision=precision.result().numpy(),
        recall=recall.result().numpy()
    )