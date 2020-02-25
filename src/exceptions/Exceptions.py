class AgentNotFoundException(Exception):
    def __init__(self):
        super(AgentNotFoundException, self).__init__()


class AgentDataNotAvailable(Exception):
    def __init__(self):
        super(AgentDataNotAvailable, self).__init__()