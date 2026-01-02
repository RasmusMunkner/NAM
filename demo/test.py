from nested_resevoirs import nested_resevoirs_hydro_api
import optax
import data

params_nr = nested_resevoirs_hydro_api.NRParameters.sample(shape=(5,))
model_nr = nested_resevoirs_hydro_api.NestedResevoirs(params=params_nr, optimizer=optax.sgd(learning_rate=0.01))

pass

final_nr, preds_nr = model_nr.predict(data.observations)