import matplotlib.pyplot as plt
import pickle

# load loss_metrics.pkl which is a list of (epoch loss, validation loss) tuples, normalize losses bewteen 0 and 1, and plot both
loss_metrics = pickle.load(open('/home/rshb/myspinner/kidney/VisibleAligned/loss_metrics.pkl', 'rb'))
loss_metrics = [(x[0]/max(loss_metrics[0]), x[1]/max(loss_metrics[1])) for x in loss_metrics]
plt.plot([x[0] for x in loss_metrics], label='Training Loss')
plt.plot([x[1] for x in loss_metrics], label='Validation Loss')
plt.legend()
plt.savefig('/home/rshb/myspinner/kidney/VisibleAligned/loss_metrics.png')
