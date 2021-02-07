def init():
    from sklearn.externals import joblib
    # load the saved model file
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_best_model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error


    