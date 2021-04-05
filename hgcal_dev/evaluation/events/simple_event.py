from hgcal_dev.evaluation.events.base_event import BaseEvent

class SimpleEvent(BaseEvent):
    def __init__(self, input_path, pred_path=None):
        super().__init__(input_path, instance_label='cluster', pred_path=pred_path, task='instance')