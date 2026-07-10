class HousePriceException(Exception):
    def __init__(self, message):
        super().__init__(message)

class DatasetNotFoundError(HousePriceException):
    pass

class DataProcessingError(HousePriceException):
    pass

class ModelTrainingError(HousePriceException):
    pass

class PredictionError(HousePriceException):
    pass
