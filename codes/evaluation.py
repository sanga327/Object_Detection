
import pandas as pd
import matplotlib.pyplot as plt


eval_df = pd.read_csv('model_checkpoint/ssd300_training_log.csv')


plt.plot(eval_df['loss'])
plt.plot(eval_df['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
plt.show()

