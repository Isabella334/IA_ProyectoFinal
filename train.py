from ml.data_loader import DataLoader
from ml.data_loader import DataLoader
from ml.model_trainer import ModelTrainer
from ml.model_exporter import ModelExporter

loader = DataLoader("./data/training_data.csv")
data = loader.load()

trainer = ModelTrainer()
trainer.train(data.X_train, data.y_train)
trainer.evaluate(data.X_test, data.y_test)

exporter = ModelExporter(trainer.get_model())
exporter.save()
exporter.validate()
