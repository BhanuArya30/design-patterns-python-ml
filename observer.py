class Observer:
    def update(self, epoch, loss):
        pass


class Logger(Observer):
    def update(self, epoch, loss):
        print(f"Epoch {epoch}, Loss: {loss}")


class Visualizer(Observer):
    def update(self, epoch, loss):
        print(f"Updating plot with loss: {loss}")


class Trainer:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, epoch, loss):
        for observer in self.observers:
            observer.update(epoch, loss)

    def train(self, epochs):
        for epoch in range(epochs):
            # Simulate loss
            loss = 0.05 * (epochs - epoch)
            self.notify(epoch, loss)


# Usage
trainer = Trainer()
trainer.attach(Logger())
trainer.attach(Visualizer())
trainer.train(5)
