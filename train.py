from ml.data_loader import DataLoader
from ml.data_loader import DataLoader
from ml.model_trainer import ModelTrainer
from ml.model_exporter import ModelExporter
from visualization.training_report import TrainingReport

loader = DataLoader("./data/training_data_sim.csv")
data = loader.load()

trainer = ModelTrainer()
trainer.train(data.X_train, data.y_train)
results = trainer.evaluate(data.X_test, data.y_test)

exporter = ModelExporter(trainer.get_model())
exporter.save()
exporter.validate()

model = trainer.get_model()
report = TrainingReport(results, list(model.classes_))
report.save()
report.show()
