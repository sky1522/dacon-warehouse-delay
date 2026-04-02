# Smart Warehouse Delay Prediction EDA

## Key Findings
- `train.csv` has 250,000 rows and 94 columns. `test.csv` has 50,000 rows and 93 columns. `layout_info.csv` provides static metadata for 300 layouts.
- The dataset follows a `12,000 scenarios x 25 slots = 300,000 rows` structure across train and test combined: train has 10,000 scenarios, test has 2,000 scenarios, and every scenario has exactly 25 rows.
- There is no explicit `timeslot` column. The slot index is implicit from row order inside each `scenario_id`, and `groupby("scenario_id").cumcount()` yields clean slots `0..24` in both train and test.
- Target `avg_delay_minutes_next_30m` is strongly right-skewed: mean `18.9623`, median `9.0327`, skew `5.6821`. IQR outliers: `14,009` rows (`5.60%`).
- Missing data is pervasive: 86 of 94 train columns contain nulls, and `99.9972%` of train rows have at least one missing value. Only `ID, layout_id, scenario_id, robot_active, robot_idle, robot_charging, robot_utilization, avg_delay_minutes_next_30m` are fully complete in train.
- Strongest target correlations are battery-related and congestion-related: `low_battery_ratio`, `battery_mean`, `robot_idle`, `order_inflow_15m`, `robot_charging`, `max_zone_density`, `battery_std`, `congestion_score`.
- `layout_info.csv` joins cleanly on `layout_id`. Train uses 250 layouts, test uses 100 layouts, and only 50 layouts overlap. That means test contains 50 layouts unseen in train, so layout metadata should be used directly.

## File Overview
|File|Rows|Cols|Dtype mix|Cols with missing|Rows with any missing|Duplicate rows|Duplicate ID|
|---|---|---|---|---|---|---|---|
|train.csv|250000|94|float64:88, str:3, int64:3|86|249993|0|0|
|test.csv|50000|93|float64:87, str:3, int64:3|86|50000|0|0|
|layout_info.csv|300|15|float64:8, int64:5, str:2|0|0|0|-|
|sample_submission.csv|50000|2|str:1, float64:1|0|0|0|0|

## train.csv
- Shape: `250,000 x 94`
- Columns (94)
```text
ID
layout_id
scenario_id
order_inflow_15m
unique_sku_15m
avg_items_per_order
urgent_order_ratio
heavy_item_ratio
cold_chain_ratio
sku_concentration
robot_active
robot_idle
robot_charging
robot_utilization
avg_trip_distance
task_reassign_15m
battery_mean
battery_std
low_battery_ratio
charge_queue_length
avg_charge_wait
congestion_score
max_zone_density
blocked_path_15m
near_collision_15m
fault_count_15m
avg_recovery_time
replenishment_overlap
pack_utilization
manual_override_ratio
avg_delay_minutes_next_30m
warehouse_temp_avg
humidity_pct
day_of_week
external_temp_c
wind_speed_kmh
precipitation_mm
lighting_level_lux
ambient_noise_db
floor_vibration_idx
return_order_ratio
air_quality_idx
co2_level_ppm
hvac_power_kw
wms_response_time_ms
scanner_error_rate
wifi_signal_db
network_latency_ms
worker_avg_tenure_months
safety_score_monthly
label_print_queue
barcode_read_success_rate
ups_battery_pct
lighting_zone_variance
shift_hour
staff_on_floor
forklift_active_count
loading_dock_util
conveyor_speed_mps
prev_shift_volume
avg_package_weight_kg
inventory_turnover_rate
daily_forecast_accuracy
order_wave_count
pick_list_length_avg
express_lane_util
bulk_order_ratio
staging_area_util
cold_storage_temp_c
pallet_wrap_time_min
fleet_age_months_avg
maintenance_schedule_score
robot_firmware_update_days
avg_idle_duration_min
charge_efficiency_pct
battery_cycle_count_avg
agv_task_success_rate
robot_calibration_score
aisle_traffic_score
zone_temp_variance
path_optimization_score
intersection_wait_time_avg
storage_density_pct
vertical_utilization
racking_height_avg_m
cross_dock_ratio
packaging_material_cost
quality_check_rate
outbound_truck_wait_min
dock_to_stock_hours
kpi_otd_pct
backorder_ratio
shift_handover_delay_min
sort_accuracy_pct
```
- Dtypes
```text
ID: str
layout_id: str
scenario_id: str
order_inflow_15m: float64
unique_sku_15m: float64
avg_items_per_order: float64
urgent_order_ratio: float64
heavy_item_ratio: float64
cold_chain_ratio: float64
sku_concentration: float64
robot_active: int64
robot_idle: int64
robot_charging: int64
robot_utilization: float64
avg_trip_distance: float64
task_reassign_15m: float64
battery_mean: float64
battery_std: float64
low_battery_ratio: float64
charge_queue_length: float64
avg_charge_wait: float64
congestion_score: float64
max_zone_density: float64
blocked_path_15m: float64
near_collision_15m: float64
fault_count_15m: float64
avg_recovery_time: float64
replenishment_overlap: float64
pack_utilization: float64
manual_override_ratio: float64
avg_delay_minutes_next_30m: float64
warehouse_temp_avg: float64
humidity_pct: float64
day_of_week: float64
external_temp_c: float64
wind_speed_kmh: float64
precipitation_mm: float64
lighting_level_lux: float64
ambient_noise_db: float64
floor_vibration_idx: float64
return_order_ratio: float64
air_quality_idx: float64
co2_level_ppm: float64
hvac_power_kw: float64
wms_response_time_ms: float64
scanner_error_rate: float64
wifi_signal_db: float64
network_latency_ms: float64
worker_avg_tenure_months: float64
safety_score_monthly: float64
label_print_queue: float64
barcode_read_success_rate: float64
ups_battery_pct: float64
lighting_zone_variance: float64
shift_hour: float64
staff_on_floor: float64
forklift_active_count: float64
loading_dock_util: float64
conveyor_speed_mps: float64
prev_shift_volume: float64
avg_package_weight_kg: float64
inventory_turnover_rate: float64
daily_forecast_accuracy: float64
order_wave_count: float64
pick_list_length_avg: float64
express_lane_util: float64
bulk_order_ratio: float64
staging_area_util: float64
cold_storage_temp_c: float64
pallet_wrap_time_min: float64
fleet_age_months_avg: float64
maintenance_schedule_score: float64
robot_firmware_update_days: float64
avg_idle_duration_min: float64
charge_efficiency_pct: float64
battery_cycle_count_avg: float64
agv_task_success_rate: float64
robot_calibration_score: float64
aisle_traffic_score: float64
zone_temp_variance: float64
path_optimization_score: float64
intersection_wait_time_avg: float64
storage_density_pct: float64
vertical_utilization: float64
racking_height_avg_m: float64
cross_dock_ratio: float64
packaging_material_cost: float64
quality_check_rate: float64
outbound_truck_wait_min: float64
dock_to_stock_hours: float64
kpi_otd_pct: float64
backorder_ratio: float64
shift_handover_delay_min: float64
sort_accuracy_pct: float64
```
- Head(10)
```csv
ID,layout_id,scenario_id,order_inflow_15m,unique_sku_15m,avg_items_per_order,urgent_order_ratio,heavy_item_ratio,cold_chain_ratio,sku_concentration,robot_active,robot_idle,robot_charging,robot_utilization,avg_trip_distance,task_reassign_15m,battery_mean,battery_std,low_battery_ratio,charge_queue_length,avg_charge_wait,congestion_score,max_zone_density,blocked_path_15m,near_collision_15m,fault_count_15m,avg_recovery_time,replenishment_overlap,pack_utilization,manual_override_ratio,avg_delay_minutes_next_30m,warehouse_temp_avg,humidity_pct,day_of_week,external_temp_c,wind_speed_kmh,precipitation_mm,lighting_level_lux,ambient_noise_db,floor_vibration_idx,return_order_ratio,air_quality_idx,co2_level_ppm,hvac_power_kw,wms_response_time_ms,scanner_error_rate,wifi_signal_db,network_latency_ms,worker_avg_tenure_months,safety_score_monthly,label_print_queue,barcode_read_success_rate,ups_battery_pct,lighting_zone_variance,shift_hour,staff_on_floor,forklift_active_count,loading_dock_util,conveyor_speed_mps,prev_shift_volume,avg_package_weight_kg,inventory_turnover_rate,daily_forecast_accuracy,order_wave_count,pick_list_length_avg,express_lane_util,bulk_order_ratio,staging_area_util,cold_storage_temp_c,pallet_wrap_time_min,fleet_age_months_avg,maintenance_schedule_score,robot_firmware_update_days,avg_idle_duration_min,charge_efficiency_pct,battery_cycle_count_avg,agv_task_success_rate,robot_calibration_score,aisle_traffic_score,zone_temp_variance,path_optimization_score,intersection_wait_time_avg,storage_density_pct,vertical_utilization,racking_height_avg_m,cross_dock_ratio,packaging_material_cost,quality_check_rate,outbound_truck_wait_min,dock_to_stock_hours,kpi_otd_pct,backorder_ratio,shift_handover_delay_min,sort_accuracy_pct
TRAIN_000000,WH_136,SC_07598,51.0,96.0,3.29,0.1176,0.1765,0.0392,0.3063,9,21,0,0.3,40.22,0.0,70.58,,0.0,0.0,0.0,0.0,0.0,0.0,,0.0,0.0,0.0867,0.3187,0.0144,5.554758292705958,,59.1,2.0,-2.8,,26.7,200.0,75.1,0.7036,0.0193,12.2,804.0,,212.0,0.0139,,40.2,84.0,80.9,17.0,0.9409,,0.9346,0.0,18.0,2.0,0.6268,,1462.0,6.72,1.33,0.7206,10.0,5.67,0.1305,0.0994,0.6683,-17.4,6.35,15.5,70.0,38.2,5.15,93.0,201.0,0.9109,94.0,24.4,1.13,77.2,1.51,0.6578,0.4437,,,4.6,0.1443,8.1,7.92,86.6,0.0787,5.12,
TRAIN_000001,WH_136,SC_07598,,93.0,2.55,0.0597,,0.0149,,12,18,0,0.4,40.96,0.0,69.87,17.6,0.0,0.0,,0.0,0.0,0.0,0.0,0.0,0.0,,0.4188,0.0144,5.03983428226648,19.52,62.7,1.0,,29.1,19.7,217.0,83.3,0.7158,0.0302,24.8,619.0,62.2,229.0,0.0066,-46.9,39.9,64.0,84.1,15.0,0.9135,74.0,1.0,4.0,,4.0,,1.49,1621.0,10.95,1.51,0.6392,11.0,7.74,0.2778,0.0651,0.5904,-24.5,5.39,21.0,70.1,58.8,3.55,90.8,247.0,,,15.8,,81.5,1.77,0.6707,0.4621,2.5,0.249,5.22,0.14,,5.48,83.9,0.085,5.77,94.88
TRAIN_000002,WH_136,SC_07598,92.0,115.0,2.49,0.0652,0.2283,0.0217,0.3063,16,14,0,0.5333,40.91,0.0,69.15,17.6,,0.0,0.0,0.0,0.0,,0.0,0.0,0.0,0.0014,0.575,0.0144,5.920881428762204,,67.1,2.0,1.2,34.1,26.7,200.0,,0.442,0.0195,10.2,834.0,60.2,338.0,,-47.1,38.2,,88.1,,0.9241,78.6,1.0,0.0,21.0,3.0,0.6566,1.52,1392.0,10.9,,0.7145,13.0,10.06,0.0958,0.0285,0.5863,-23.0,5.97,16.0,70.4,42.2,2.9,88.1,226.0,0.8966,86.1,38.0,2.16,64.5,2.06,0.7378,,3.87,0.1977,4.26,0.1817,10.7,6.88,82.1,0.1052,,94.4
TRAIN_000003,WH_136,SC_07598,77.0,110.0,2.52,0.0649,,0.039,0.3063,13,17,0,0.4333,40.38,0.0,68.43,17.6,0.0,0.0,0.0,0.0,,0.0,0.0,0.0,0.0,0.0406,0.4813,0.0144,5.540926247519362,18.73,51.7,3.0,-0.5,33.5,23.7,261.0,79.0,0.5258,0.0564,3.9,874.0,63.9,291.0,0.0192,,33.2,97.0,86.2,,0.9099,74.4,0.8267,6.0,28.0,,0.6594,1.51,2176.0,13.13,1.67,0.6516,8.0,11.53,0.1415,0.0576,0.6972,-22.6,6.13,24.4,70.2,,2.58,91.6,,,93.9,18.6,1.28,64.6,1.08,0.6277,0.3741,2.5,0.1955,4.89,0.1485,10.7,6.76,87.9,0.092,4.53,93.72
TRAIN_000004,WH_136,SC_07598,,122.0,3.12,0.0667,0.3333,,0.3063,10,20,0,0.3333,39.8,0.0,67.71,17.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0082,0.375,0.0144,3.9400706320306647,20.51,55.6,1.0,-5.0,,21.1,221.0,,0.5129,0.0366,5.8,891.0,80.0,228.0,0.0106,-57.3,,,88.6,18.0,0.9272,82.4,1.0,5.0,,,0.6422,1.74,,13.66,1.77,0.7244,6.0,8.05,0.1868,,0.4822,-23.9,5.82,22.9,71.6,56.5,3.25,90.4,205.0,0.8843,96.8,22.7,1.38,73.6,0.34,0.6834,,,0.2351,5.16,0.1514,12.4,9.03,83.8,0.0843,3.99,95.02
TRAIN_000005,WH_136,SC_07598,59.0,,2.53,0.0847,0.2881,0.0169,0.3063,10,20,0,0.3333,39.99,0.0,66.99,17.6,0.0,0.0,0.0,0.0,,0.0,,0.0,0.0,0.0285,0.3688,0.0144,6.896073640520525,19.06,60.4,,-1.3,34.0,24.8,200.0,70.6,,0.0039,6.8,832.0,49.5,285.0,0.0071,-46.4,36.5,78.0,78.0,14.0,,,1.0,6.0,,4.0,0.6587,1.56,1459.0,11.37,2.56,0.7449,8.0,8.02,0.1604,0.0774,0.5987,,5.69,20.2,71.7,43.7,4.7,88.2,252.0,0.9172,92.1,32.1,,67.9,2.43,0.6885,0.527,3.82,0.0861,6.65,0.1175,9.8,8.55,83.5,0.0847,2.73,96.35
TRAIN_000006,WH_136,SC_07598,53.0,,,,0.2642,,,9,21,0,0.3,,0.0,66.27,17.6,0.0,0.0,,0.0,0.0,0.0,0.0,0.0,0.0,0.0214,0.3312,0.0144,6.069856925697026,22.15,71.1,3.0,-3.3,22.4,19.0,316.0,,0.4259,0.035,9.2,882.0,60.9,245.0,0.0078,-46.6,25.0,72.0,81.7,8.0,,77.6,0.8796,6.0,20.0,4.0,0.4873,1.89,1876.0,13.08,1.19,0.6825,8.0,6.93,0.1385,0.0343,0.5979,,4.71,18.3,75.3,58.8,2.7,88.7,181.0,0.8809,95.8,32.1,1.32,79.6,1.5,,0.5747,,0.1574,3.93,0.1662,6.6,5.84,86.9,0.0751,4.25,93.5
TRAIN_000007,WH_136,SC_07598,,125.0,2.42,0.0674,0.2472,0.0449,0.3063,15,15,0,0.5,39.82,0.0,65.55,17.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0981,0.5563,,5.266311630532883,19.54,59.3,2.0,6.8,,,221.0,79.6,0.579,0.0464,,971.0,68.7,249.0,0.0173,-43.4,35.7,74.0,87.0,13.0,0.9296,83.6,0.8646,0.0,20.0,3.0,0.5679,1.52,2157.0,12.87,1.38,0.742,10.0,7.75,0.1825,0.027,0.4971,-22.1,,19.8,,47.0,1.59,89.9,171.0,0.8766,90.8,27.6,2.03,71.6,1.61,0.6262,0.5643,,0.1456,5.44,0.1817,11.0,7.74,84.1,0.1155,,
TRAIN_000008,WH_136,SC_07598,94.0,135.0,2.35,0.0851,0.2447,0.1064,0.3063,16,14,0,0.5333,40.55,0.0,64.83,17.6,0.0,,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0381,0.5875,0.0144,5.819306523426223,,63.5,2.0,-5.0,,17.4,261.0,78.5,,0.0594,30.3,713.0,66.5,266.0,0.0187,-55.0,38.6,96.0,87.3,13.0,0.9039,84.2,0.8517,6.0,21.0,,0.6123,1.22,1525.0,9.99,2.42,0.7146,,,0.2685,0.0254,0.6823,-13.6,5.96,24.7,,48.9,3.55,84.5,216.0,0.8997,91.3,31.1,1.85,78.4,1.56,0.796,0.5331,2.78,0.1701,,0.1633,,3.15,85.5,0.0516,4.25,94.27
TRAIN_000009,WH_136,SC_07598,78.0,,2.46,0.0769,0.2564,0.0769,0.3063,13,17,0,0.4333,40.86,0.0,64.12,,,0.0,0.0,,0.0,0.0,0.0,,0.0,0.0068,0.4875,0.0144,5.83424496313056,19.7,59.7,4.0,9.8,30.4,25.6,200.0,79.2,,0.0532,13.4,817.0,66.2,249.0,0.0117,-48.5,32.3,98.0,77.9,,0.9006,84.2,0.8472,7.0,22.0,4.0,,1.28,2306.0,12.98,,0.6333,12.0,4.61,0.2673,0.0395,,-22.5,5.22,11.9,72.0,58.8,4.79,92.2,192.0,0.8905,96.7,25.9,2.07,69.5,0.99,0.7152,,6.12,0.1188,6.58,0.159,8.2,,,0.0868,6.38,94.62
```

## test.csv
- Shape: `50,000 x 93`
- Columns (93)
```text
ID
layout_id
scenario_id
order_inflow_15m
unique_sku_15m
avg_items_per_order
urgent_order_ratio
heavy_item_ratio
cold_chain_ratio
sku_concentration
robot_active
robot_idle
robot_charging
robot_utilization
avg_trip_distance
task_reassign_15m
battery_mean
battery_std
low_battery_ratio
charge_queue_length
avg_charge_wait
congestion_score
max_zone_density
blocked_path_15m
near_collision_15m
fault_count_15m
avg_recovery_time
replenishment_overlap
pack_utilization
manual_override_ratio
warehouse_temp_avg
humidity_pct
day_of_week
external_temp_c
wind_speed_kmh
precipitation_mm
lighting_level_lux
ambient_noise_db
floor_vibration_idx
return_order_ratio
air_quality_idx
co2_level_ppm
hvac_power_kw
wms_response_time_ms
scanner_error_rate
wifi_signal_db
network_latency_ms
worker_avg_tenure_months
safety_score_monthly
label_print_queue
barcode_read_success_rate
ups_battery_pct
lighting_zone_variance
shift_hour
staff_on_floor
forklift_active_count
loading_dock_util
conveyor_speed_mps
prev_shift_volume
avg_package_weight_kg
inventory_turnover_rate
daily_forecast_accuracy
order_wave_count
pick_list_length_avg
express_lane_util
bulk_order_ratio
staging_area_util
cold_storage_temp_c
pallet_wrap_time_min
fleet_age_months_avg
maintenance_schedule_score
robot_firmware_update_days
avg_idle_duration_min
charge_efficiency_pct
battery_cycle_count_avg
agv_task_success_rate
robot_calibration_score
aisle_traffic_score
zone_temp_variance
path_optimization_score
intersection_wait_time_avg
storage_density_pct
vertical_utilization
racking_height_avg_m
cross_dock_ratio
packaging_material_cost
quality_check_rate
outbound_truck_wait_min
dock_to_stock_hours
kpi_otd_pct
backorder_ratio
shift_handover_delay_min
sort_accuracy_pct
```
- Dtypes
```text
ID: str
layout_id: str
scenario_id: str
order_inflow_15m: float64
unique_sku_15m: float64
avg_items_per_order: float64
urgent_order_ratio: float64
heavy_item_ratio: float64
cold_chain_ratio: float64
sku_concentration: float64
robot_active: int64
robot_idle: int64
robot_charging: int64
robot_utilization: float64
avg_trip_distance: float64
task_reassign_15m: float64
battery_mean: float64
battery_std: float64
low_battery_ratio: float64
charge_queue_length: float64
avg_charge_wait: float64
congestion_score: float64
max_zone_density: float64
blocked_path_15m: float64
near_collision_15m: float64
fault_count_15m: float64
avg_recovery_time: float64
replenishment_overlap: float64
pack_utilization: float64
manual_override_ratio: float64
warehouse_temp_avg: float64
humidity_pct: float64
day_of_week: float64
external_temp_c: float64
wind_speed_kmh: float64
precipitation_mm: float64
lighting_level_lux: float64
ambient_noise_db: float64
floor_vibration_idx: float64
return_order_ratio: float64
air_quality_idx: float64
co2_level_ppm: float64
hvac_power_kw: float64
wms_response_time_ms: float64
scanner_error_rate: float64
wifi_signal_db: float64
network_latency_ms: float64
worker_avg_tenure_months: float64
safety_score_monthly: float64
label_print_queue: float64
barcode_read_success_rate: float64
ups_battery_pct: float64
lighting_zone_variance: float64
shift_hour: float64
staff_on_floor: float64
forklift_active_count: float64
loading_dock_util: float64
conveyor_speed_mps: float64
prev_shift_volume: float64
avg_package_weight_kg: float64
inventory_turnover_rate: float64
daily_forecast_accuracy: float64
order_wave_count: float64
pick_list_length_avg: float64
express_lane_util: float64
bulk_order_ratio: float64
staging_area_util: float64
cold_storage_temp_c: float64
pallet_wrap_time_min: float64
fleet_age_months_avg: float64
maintenance_schedule_score: float64
robot_firmware_update_days: float64
avg_idle_duration_min: float64
charge_efficiency_pct: float64
battery_cycle_count_avg: float64
agv_task_success_rate: float64
robot_calibration_score: float64
aisle_traffic_score: float64
zone_temp_variance: float64
path_optimization_score: float64
intersection_wait_time_avg: float64
storage_density_pct: float64
vertical_utilization: float64
racking_height_avg_m: float64
cross_dock_ratio: float64
packaging_material_cost: float64
quality_check_rate: float64
outbound_truck_wait_min: float64
dock_to_stock_hours: float64
kpi_otd_pct: float64
backorder_ratio: float64
shift_handover_delay_min: float64
sort_accuracy_pct: float64
```
- Head(10)
```csv
ID,layout_id,scenario_id,order_inflow_15m,unique_sku_15m,avg_items_per_order,urgent_order_ratio,heavy_item_ratio,cold_chain_ratio,sku_concentration,robot_active,robot_idle,robot_charging,robot_utilization,avg_trip_distance,task_reassign_15m,battery_mean,battery_std,low_battery_ratio,charge_queue_length,avg_charge_wait,congestion_score,max_zone_density,blocked_path_15m,near_collision_15m,fault_count_15m,avg_recovery_time,replenishment_overlap,pack_utilization,manual_override_ratio,warehouse_temp_avg,humidity_pct,day_of_week,external_temp_c,wind_speed_kmh,precipitation_mm,lighting_level_lux,ambient_noise_db,floor_vibration_idx,return_order_ratio,air_quality_idx,co2_level_ppm,hvac_power_kw,wms_response_time_ms,scanner_error_rate,wifi_signal_db,network_latency_ms,worker_avg_tenure_months,safety_score_monthly,label_print_queue,barcode_read_success_rate,ups_battery_pct,lighting_zone_variance,shift_hour,staff_on_floor,forklift_active_count,loading_dock_util,conveyor_speed_mps,prev_shift_volume,avg_package_weight_kg,inventory_turnover_rate,daily_forecast_accuracy,order_wave_count,pick_list_length_avg,express_lane_util,bulk_order_ratio,staging_area_util,cold_storage_temp_c,pallet_wrap_time_min,fleet_age_months_avg,maintenance_schedule_score,robot_firmware_update_days,avg_idle_duration_min,charge_efficiency_pct,battery_cycle_count_avg,agv_task_success_rate,robot_calibration_score,aisle_traffic_score,zone_temp_variance,path_optimization_score,intersection_wait_time_avg,storage_density_pct,vertical_utilization,racking_height_avg_m,cross_dock_ratio,packaging_material_cost,quality_check_rate,outbound_truck_wait_min,dock_to_stock_hours,kpi_otd_pct,backorder_ratio,shift_handover_delay_min,sort_accuracy_pct
TEST_000000,WH_202,SC_10973,61.0,155.0,,0.0,0.0656,0.1311,0.3047,11,31,2,0.25,44.26,0.0,,,0.0,0.0,0.0,16.7,0.0682,0.0,0.0,2.0,1.87,0.0,0.0098,0.0841,16.67,60.6,5.0,19.6,18.5,,421.0,90.0,,0.0922,0.0,684.0,74.5,,0.0145,,43.6,91.0,75.6,12.0,0.9954,63.9,0.7037,16.0,19.0,5.0,0.5099,1.66,1917.0,4.07,1.86,,,12.61,0.0749,0.2521,0.6405,-7.3,2.6,45.9,,41.2,7.03,91.0,98.0,0.9334,96.4,47.7,1.8,75.5,1.63,0.437,0.6587,4.88,0.0534,4.93,0.2164,13.5,,96.2,0.0374,0.0,97.74
TEST_000001,WH_202,SC_10973,59.0,108.0,,,0.0847,0.0847,0.3047,20,24,0,0.4545,,0.0,66.59,17.45,0.0,0.0,0.0,20.04,0.0909,2.0,0.0,0.0,0.0,0.0,0.4706,0.0869,23.38,47.1,3.0,,25.2,17.6,498.0,90.0,0.4479,0.1101,0.0,562.0,63.8,50.0,,-39.8,50.0,76.0,68.7,15.0,0.9928,72.4,0.769,16.0,22.0,5.0,0.6409,1.66,2221.0,3.56,2.51,0.8653,20.0,11.2,0.0376,,0.4746,-8.6,3.31,45.9,70.0,36.0,7.45,94.8,80.0,,87.7,45.5,0.91,76.1,2.39,0.5515,,6.69,,,,21.1,15.94,90.5,0.0522,3.03,
TEST_000002,WH_202,SC_10973,95.0,200.0,3.35,0.0526,,0.1053,0.3047,28,15,1,0.6364,43.29,0.0,63.92,17.56,0.0,0.0,0.0,25.05,0.1136,0.0,2.0,1.0,4.81,0.0517,0.7549,0.0912,19.88,72.7,,11.2,,16.9,477.0,86.1,0.2523,0.091,29.0,530.0,,112.0,0.0078,-30.0,46.9,89.0,75.8,14.0,0.9885,,0.5856,16.0,34.0,5.0,0.6232,1.36,2219.0,5.81,2.04,0.882,19.0,,0.0822,0.1913,0.5909,-6.9,2.38,46.0,70.7,42.8,8.98,98.5,61.0,0.9527,87.7,50.3,1.91,91.8,,0.4588,0.6956,6.28,0.0534,8.57,0.2292,23.2,16.25,,0.0649,1.36,95.31
TEST_000003,WH_202,SC_10973,59.0,166.0,4.2,0.0339,0.0508,0.1017,0.3047,25,18,1,0.5682,43.29,0.0,61.89,17.29,0.0,0.0,0.0,16.7,0.1136,1.0,3.0,1.0,11.53,0.1145,0.8627,0.0841,19.09,65.5,4.0,7.8,18.6,18.3,382.0,87.0,0.3113,,8.7,528.0,46.9,131.0,,-31.5,38.1,,84.7,12.0,0.9732,67.4,0.672,16.0,26.0,4.0,0.5217,1.11,1371.0,2.17,2.74,0.886,18.0,12.83,0.1553,0.217,0.6042,-3.1,3.73,,73.0,33.7,7.57,88.7,81.0,0.9287,90.1,54.2,1.21,78.7,1.03,0.4589,,7.92,0.0835,,0.2045,,15.57,,0.0268,0.64,94.01
TEST_000004,WH_202,SC_10973,,130.0,,0.02,0.02,0.1,0.3047,19,25,0,0.4318,43.5,0.0,60.12,17.36,0.0227,0.0,0.0,15.03,0.0909,1.0,2.0,0.0,0.0,0.0,,0.0827,18.71,67.6,4.0,13.1,16.7,14.7,325.0,86.7,0.3921,0.1405,11.1,600.0,54.0,99.0,0.0073,,48.6,86.0,74.8,15.0,0.9821,57.1,0.8801,17.0,16.0,4.0,0.65,1.17,1611.0,2.13,2.43,0.9735,,13.06,,0.1984,,-6.3,3.2,45.8,78.6,35.7,,89.2,50.0,0.9467,95.5,47.6,2.48,80.4,0.87,,0.7176,7.0,0.1684,,0.2174,,17.16,91.0,0.0533,0.25,95.66
TEST_000005,WH_202,SC_10973,,83.0,4.87,0.1,0.1333,0.0667,0.3047,14,29,1,0.3182,42.18,0.0,,17.81,0.0227,0.0,0.0,8.35,,0.0,1.0,0.0,0.0,0.0124,,0.0771,20.8,77.6,4.0,24.0,22.2,18.2,525.0,90.0,0.4522,,10.8,559.0,50.5,,0.0139,-41.8,45.0,95.0,75.3,12.0,,61.2,0.5767,17.0,25.0,4.0,0.6319,0.98,1725.0,3.24,,0.9843,19.0,12.31,0.1514,0.2492,,-7.7,3.47,43.0,79.3,27.6,8.67,91.1,96.0,0.9137,86.0,52.2,1.29,79.5,0.69,0.4031,0.7817,5.8,0.0778,7.64,0.2175,16.4,12.41,95.3,0.0665,0.0,95.75
TEST_000006,WH_202,SC_10973,25.0,53.0,4.4,0.04,0.04,0.04,0.3047,10,33,1,0.2273,41.96,,58.19,17.56,0.0,0.0,0.0,6.68,0.0682,0.0,0.0,0.0,0.0,0.0079,0.3039,0.0757,19.33,65.0,4.0,14.8,,20.2,374.0,86.9,0.5384,0.0869,0.0,679.0,73.9,50.0,0.0167,-37.5,38.2,99.0,74.2,11.0,0.9979,57.3,0.5086,16.0,,3.0,0.6111,1.28,2162.0,0.98,2.14,0.9411,19.0,9.54,0.1804,0.2982,0.4509,-8.2,,46.0,79.4,,9.15,89.2,50.0,0.9188,90.4,,2.02,81.7,0.67,0.4031,0.7817,,0.0534,6.59,0.2209,,15.59,,0.062,0.0,98.34
TEST_000007,WH_202,SC_10973,48.0,125.0,4.19,0.0208,0.125,0.0833,0.3047,12,31,1,0.2727,42.8,0.0,56.99,17.57,,0.0,0.0,13.36,0.0455,0.0,0.0,0.0,0.0,,0.2353,0.0813,15.84,53.0,4.0,9.5,19.7,14.0,,90.0,0.5857,0.0597,0.0,,64.9,66.0,0.0273,-32.2,44.6,101.0,74.9,11.0,0.9973,,,17.0,26.0,5.0,0.6068,1.9,1504.0,2.73,1.53,0.8612,19.0,11.13,0.0661,0.3247,0.6223,-6.6,3.57,43.9,78.2,18.1,7.94,93.6,74.0,,93.2,33.5,,82.2,2.43,0.5011,,5.13,0.0869,5.19,0.2245,18.4,14.41,96.8,0.0338,1.42,97.89
TEST_000008,WH_202,SC_10973,53.0,118.0,3.79,0.0189,0.0377,,0.3047,17,25,2,0.3864,43.09,0.0,55.58,,0.0455,,0.0,15.03,0.0682,0.0,1.0,0.0,0.0,0.0436,0.4706,0.0827,18.21,70.3,4.0,19.7,15.3,17.4,466.0,82.2,0.617,0.0806,0.0,541.0,67.0,110.0,0.0084,,45.6,99.0,,13.0,1.0,58.8,0.6596,10.0,20.0,3.0,0.608,1.03,2306.0,2.05,1.49,0.9394,20.0,,0.021,0.2076,0.6174,-6.6,2.87,37.7,69.0,41.5,10.99,94.6,96.0,,84.0,43.8,2.18,,2.63,0.4557,0.7005,6.27,0.0534,6.23,0.2254,14.3,,87.8,0.0394,0.55,95.8
TEST_000009,WH_202,SC_10973,100.0,231.0,3.54,,0.06,0.1,0.3047,26,14,4,0.5909,43.73,0.0,53.66,17.38,0.0227,0.0,0.0,25.05,0.1818,5.0,3.0,1.0,1.21,0.0114,0.6176,0.0912,18.96,58.4,3.0,21.5,14.2,14.9,359.0,83.0,0.1849,0.1324,0.0,806.0,80.0,63.0,0.0159,-34.2,,91.0,,12.0,1.0,54.2,0.6648,16.0,23.0,5.0,0.5681,0.92,2109.0,,2.0,0.8444,19.0,9.43,0.0333,0.2172,0.5748,-4.4,2.99,36.0,74.3,23.0,9.5,90.8,50.0,0.9117,87.9,49.6,2.39,,4.03,0.5299,0.7817,6.77,0.0534,7.89,,19.7,17.97,92.8,0.0473,0.0,95.96
```

## layout_info.csv
- Shape: `300 x 15`
- Columns (15)
```text
layout_id
layout_type
aisle_width_avg
intersection_count
one_way_ratio
pack_station_count
charger_count
layout_compactness
zone_dispersion
robot_total
building_age_years
floor_area_sqm
ceiling_height_m
fire_sprinkler_count
emergency_exit_count
```
- Dtypes
```text
layout_id: str
layout_type: str
aisle_width_avg: float64
intersection_count: float64
one_way_ratio: float64
pack_station_count: float64
charger_count: float64
layout_compactness: float64
zone_dispersion: float64
robot_total: int64
building_age_years: int64
floor_area_sqm: int64
ceiling_height_m: float64
fire_sprinkler_count: int64
emergency_exit_count: int64
```
- Head(10)
```csv
layout_id,layout_type,aisle_width_avg,intersection_count,one_way_ratio,pack_station_count,charger_count,layout_compactness,zone_dispersion,robot_total,building_age_years,floor_area_sqm,ceiling_height_m,fire_sprinkler_count,emergency_exit_count
WH_001,narrow,2.08,34.0,0.3874,9.0,8.0,0.8078,0.5867,21,20,3384,9.4,36,5
WH_002,grid,3.7,16.0,0.0054,9.0,11.0,0.7339,0.439,57,26,8311,8.1,12,2
WH_003,grid,2.54,52.0,0.0229,5.0,3.0,0.5498,0.3708,63,39,9465,7.2,81,2
WH_004,hybrid,3.37,35.0,0.4836,5.0,10.0,0.4821,0.5661,100,40,7918,12.7,26,5
WH_005,grid,3.68,48.0,0.0494,8.0,13.0,0.674,0.997,50,33,5905,5.6,84,9
WH_006,grid,3.03,20.0,0.0902,6.0,10.0,0.7925,0.2971,33,45,8425,4.3,11,7
WH_007,narrow,1.6,33.0,0.5534,19.0,15.0,0.8665,0.3583,18,38,9967,12.6,22,4
WH_008,narrow,1.99,19.0,0.5355,13.0,12.0,0.9702,0.2661,74,27,9551,10.4,58,2
WH_009,grid,3.08,28.0,0.1392,7.0,6.0,0.5615,0.9592,25,22,6085,9.6,89,2
WH_010,hub_spoke,3.02,20.0,0.1229,8.0,9.0,0.5389,0.6837,13,48,8538,13.2,58,10
```

## sample_submission.csv
- Shape: `50,000 x 2`
- Columns (2)
```text
ID
avg_delay_minutes_next_30m
```
- Dtypes
```text
ID: str
avg_delay_minutes_next_30m: float64
```
- Head(10)
```csv
ID,avg_delay_minutes_next_30m
TEST_000000,0.0
TEST_000001,0.0
TEST_000002,0.0
TEST_000003,0.0
TEST_000004,0.0
TEST_000005,0.0
TEST_000006,0.0
TEST_000007,0.0
TEST_000008,0.0
TEST_000009,0.0
```

## 90 Train Features and Name-Based Meaning Estimates
- Definition: all numeric predictors in `train.csv`, excluding identifiers (`ID`, `layout_id`, `scenario_id`) and the target.
### Order and demand (15)
|Feature|Estimated meaning|
|---|---|
|order_inflow_15m|Order inflow over the last 15 minutes|
|unique_sku_15m|Number of unique SKUs over the last 15 minutes|
|avg_items_per_order|Average item count per order|
|urgent_order_ratio|Share of urgent orders|
|heavy_item_ratio|Share of heavy-item orders|
|cold_chain_ratio|Share of cold-chain orders|
|sku_concentration|How concentrated demand is among top SKUs|
|return_order_ratio|Share of return orders|
|avg_package_weight_kg|Average package weight|
|inventory_turnover_rate|Inventory turnover rate|
|daily_forecast_accuracy|Daily demand forecast accuracy|
|order_wave_count|Number of order waves|
|pick_list_length_avg|Average pick-list length|
|bulk_order_ratio|Share of bulk orders|
|backorder_ratio|Backorder ratio|

### Robot operations (9)
|Feature|Estimated meaning|
|---|---|
|robot_active|Number of robots actively working|
|robot_idle|Number of idle robots|
|robot_utilization|Robot utilization rate|
|avg_trip_distance|Average robot travel distance per task|
|task_reassign_15m|Task reassignment amount in the last 15 minutes|
|avg_idle_duration_min|Average robot idle duration|
|agv_task_success_rate|AGV task success rate|
|robot_calibration_score|Robot calibration score|
|path_optimization_score|Path optimization score|

### Battery and charging (8)
|Feature|Estimated meaning|
|---|---|
|robot_charging|Number of robots currently charging|
|battery_mean|Mean robot battery level|
|battery_std|Dispersion of robot battery levels|
|low_battery_ratio|Share of low-battery robots|
|charge_queue_length|Charging queue length|
|avg_charge_wait|Average charging wait time|
|charge_efficiency_pct|Charging efficiency|
|battery_cycle_count_avg|Average battery cycle count|

### Congestion and safety (8)
|Feature|Estimated meaning|
|---|---|
|congestion_score|Overall warehouse congestion score|
|max_zone_density|Highest zone density in the warehouse|
|blocked_path_15m|Path blockage count in the last 15 minutes|
|near_collision_15m|Near-collision count in the last 15 minutes|
|fault_count_15m|Fault count in the last 15 minutes|
|avg_recovery_time|Average time to recover from a fault|
|aisle_traffic_score|Aisle traffic score|
|intersection_wait_time_avg|Average wait time at intersections|

### Packing and outbound (16)
|Feature|Estimated meaning|
|---|---|
|pack_utilization|Packing station utilization|
|label_print_queue|Label printing queue length|
|barcode_read_success_rate|Barcode read success rate|
|forklift_active_count|Number of active forklifts|
|loading_dock_util|Loading dock utilization|
|conveyor_speed_mps|Conveyor speed in meters per second|
|express_lane_util|Express lane utilization|
|staging_area_util|Staging area utilization|
|pallet_wrap_time_min|Pallet wrapping time|
|cross_dock_ratio|Share of cross-dock processing|
|packaging_material_cost|Packaging material cost|
|quality_check_rate|Quality check rate|
|outbound_truck_wait_min|Outbound truck wait time|
|dock_to_stock_hours|Hours from dock to stock|
|kpi_otd_pct|On-time delivery KPI percentage|
|sort_accuracy_pct|Sort accuracy|

### Environment and facility (15)
|Feature|Estimated meaning|
|---|---|
|warehouse_temp_avg|Average warehouse temperature|
|humidity_pct|Humidity percentage|
|external_temp_c|External temperature|
|wind_speed_kmh|Wind speed|
|precipitation_mm|Precipitation amount|
|lighting_level_lux|Lighting level|
|ambient_noise_db|Ambient noise level|
|floor_vibration_idx|Floor vibration index|
|air_quality_idx|Indoor air quality index|
|co2_level_ppm|CO2 concentration|
|hvac_power_kw|HVAC power usage|
|ups_battery_pct|UPS battery level|
|lighting_zone_variance|Variance in lighting across zones|
|cold_storage_temp_c|Cold storage temperature|
|zone_temp_variance|Temperature variance across zones|

### IT and systems (4)
|Feature|Estimated meaning|
|---|---|
|wms_response_time_ms|WMS response time|
|scanner_error_rate|Scanner error rate|
|wifi_signal_db|Wi-Fi signal strength|
|network_latency_ms|Network latency|

### Workforce (4)
|Feature|Estimated meaning|
|---|---|
|worker_avg_tenure_months|Average worker tenure in months|
|safety_score_monthly|Monthly safety score|
|staff_on_floor|Number of workers on the floor|
|shift_handover_delay_min|Shift handover delay|

### Ops context (5)
|Feature|Estimated meaning|
|---|---|
|replenishment_overlap|Overlap between replenishment work and outbound work|
|manual_override_ratio|Share of tasks requiring manual override|
|day_of_week|Day-of-week index|
|shift_hour|Shift hour / time-of-day indicator|
|prev_shift_volume|Volume processed in the previous shift|

### Maintenance and assets (3)
|Feature|Estimated meaning|
|---|---|
|fleet_age_months_avg|Average asset/fleet age in months|
|maintenance_schedule_score|Maintenance schedule quality score|
|robot_firmware_update_days|Days since robot firmware update|

### Space and layout (3)
|Feature|Estimated meaning|
|---|---|
|storage_density_pct|Storage density|
|vertical_utilization|Vertical space utilization|
|racking_height_avg_m|Average rack height|

## Target Distribution
|Metric|Value|
|---|---|
|count|250,000|
|mean|18.9623|
|median|9.0327|
|std|27.3514|
|skew|5.6821|
|min|0.0000|
|q1|4.2788|
|q3|25.7919|
|q90|45.2437|
|q95|60.7927|
|q99|120.8547|
|max|715.8581|
|zero count|6,817|
|zero ratio|2.7268%|
|IQR lower fence|-27.9908|
|IQR upper fence|58.0615|
|IQR outlier count|14,009|
|IQR outlier ratio|5.6036%|

- Mean is about 2.1x the median, confirming a heavy right tail.
- The 99th percentile is `120.8547` while the maximum is `715.8581`, so there are extreme delay events.
- `log1p(target)`, Huber-style objectives, quantile losses, or robust tree models are reasonable follow-up experiments.

## Missing Values and Duplicates
- Fully complete train columns: `ID, layout_id, scenario_id, robot_active, robot_idle, robot_charging, robot_utilization, avg_delay_minutes_next_30m`
- Fully complete test columns: `ID, layout_id, scenario_id, robot_active, robot_idle, robot_charging, robot_utilization`
- No fully duplicated rows were found in any file. `ID` is unique in train, test, and sample submission.

### Top 20 missing columns in train
|Column|Missing count|Missing ratio|
|---|---|---|
|avg_recovery_time|32,529|13.0116%|
|congestion_score|32,250|12.9000%|
|avg_charge_wait|30,696|12.2784%|
|battery_mean|30,320|12.1280%|
|charge_efficiency_pct|30,052|12.0208%|
|battery_cycle_count_avg|29,955|11.9820%|
|fleet_age_months_avg|29,953|11.9812%|
|robot_calibration_score|29,944|11.9776%|
|unique_sku_15m|29,924|11.9696%|
|staging_area_util|29,892|11.9568%|
|humidity_pct|29,881|11.9524%|
|barcode_read_success_rate|29,862|11.9448%|
|inventory_turnover_rate|29,853|11.9412%|
|worker_avg_tenure_months|29,804|11.9216%|
|shift_hour|29,797|11.9188%|
|sort_accuracy_pct|29,770|11.9080%|
|safety_score_monthly|29,766|11.9064%|
|avg_trip_distance|29,766|11.9064%|
|urgent_order_ratio|29,754|11.9016%|
|return_order_ratio|29,732|11.8928%|

### Top 20 missing columns in test
|Column|Missing count|Missing ratio|
|---|---|---|
|avg_recovery_time|6,676|13.3520%|
|congestion_score|6,318|12.6360%|
|avg_charge_wait|6,199|12.3980%|
|battery_cycle_count_avg|6,135|12.2700%|
|wifi_signal_db|6,090|12.1800%|
|robot_firmware_update_days|6,075|12.1500%|
|external_temp_c|6,072|12.1440%|
|pallet_wrap_time_min|6,053|12.1060%|
|battery_mean|6,051|12.1020%|
|barcode_read_success_rate|6,048|12.0960%|
|cold_storage_temp_c|6,046|12.0920%|
|inventory_turnover_rate|6,040|12.0800%|
|unique_sku_15m|6,032|12.0640%|
|bulk_order_ratio|6,030|12.0600%|
|vertical_utilization|6,029|12.0580%|
|co2_level_ppm|6,021|12.0420%|
|fault_count_15m|6,015|12.0300%|
|hvac_power_kw|6,010|12.0200%|
|network_latency_ms|6,008|12.0160%|
|charge_queue_length|6,004|12.0080%|

- Missingness ratios are remarkably uniform across many columns (roughly 11.8% to 13.4%), which suggests scenario-level or sensor-batch missingness rather than a single broken field.
- Do not drop missing rows. Add missing-indicator features and compare global median imputation with scenario-level or layout-level grouped imputation.

## Top 20 Feature Correlations with the Target
|Rank|Feature|Corr|Abs corr|Category|Estimated meaning|
|---|---|---|---|---|---|
|1|low_battery_ratio|0.3661|0.3661|Battery and charging|Share of low-battery robots|
|2|battery_mean|-0.3589|0.3589|Battery and charging|Mean robot battery level|
|3|robot_idle|-0.3493|0.3493|Robot operations|Number of idle robots|
|4|order_inflow_15m|0.3418|0.3418|Order and demand|Order inflow over the last 15 minutes|
|5|robot_charging|0.3204|0.3204|Battery and charging|Number of robots currently charging|
|6|max_zone_density|0.3108|0.3108|Congestion and safety|Highest zone density in the warehouse|
|7|battery_std|0.3083|0.3083|Battery and charging|Dispersion of robot battery levels|
|8|congestion_score|0.3004|0.3004|Congestion and safety|Overall warehouse congestion score|
|9|sku_concentration|0.2919|0.2919|Order and demand|How concentrated demand is among top SKUs|
|10|urgent_order_ratio|0.2711|0.2711|Order and demand|Share of urgent orders|
|11|charge_queue_length|0.2614|0.2614|Battery and charging|Charging queue length|
|12|avg_charge_wait|0.2515|0.2515|Battery and charging|Average charging wait time|
|13|near_collision_15m|0.2428|0.2428|Congestion and safety|Near-collision count in the last 15 minutes|
|14|unique_sku_15m|0.2290|0.2290|Order and demand|Number of unique SKUs over the last 15 minutes|
|15|blocked_path_15m|0.2205|0.2205|Congestion and safety|Path blockage count in the last 15 minutes|
|16|loading_dock_util|0.2131|0.2131|Packing and outbound|Loading dock utilization|
|17|robot_utilization|0.2108|0.2108|Robot operations|Robot utilization rate|
|18|heavy_item_ratio|0.2103|0.2103|Order and demand|Share of heavy-item orders|
|19|fault_count_15m|0.2029|0.2029|Congestion and safety|Fault count in the last 15 minutes|
|20|maintenance_schedule_score|-0.1968|0.1968|Maintenance and assets|Maintenance schedule quality score|

- Positive correlations point to load and bottlenecks: higher order inflow, congestion, charging queues, density, and near-collisions all align with longer delays.
- Negative correlations point to slack capacity and better health: higher `battery_mean`, more `robot_idle`, and higher `maintenance_schedule_score` align with shorter delays.
- The largest absolute correlation is only about `0.366`, so interactions, sequence context, and grouped structure likely matter more than any single raw feature.

## layout_info Join Key and Recommended Usage
- Common join key: `layout_id`
- Unique `layout_id`: train `250`, test `100`, layout_info `300`
- Train layouts missing in `layout_info`: `0`
- Each `scenario_id` maps to exactly one `layout_id` in both train and test.
- Layout overlap between train and test: `50`. Test-only layouts: `50`.

### layout_info column meanings
|Column|Meaning|
|---|---|
|layout_type|Layout archetype (grid, hub_spoke, hybrid, narrow)|
|aisle_width_avg|Average aisle width|
|intersection_count|Number of intersections|
|one_way_ratio|Share of one-way aisles|
|pack_station_count|Number of packing stations|
|charger_count|Number of chargers|
|layout_compactness|Compactness of the layout|
|zone_dispersion|How dispersed warehouse zones are|
|robot_total|Total robot count|
|building_age_years|Building age in years|
|floor_area_sqm|Floor area in square meters|
|ceiling_height_m|Ceiling height|
|fire_sprinkler_count|Fire sprinkler count|
|emergency_exit_count|Emergency exit count|

### Top layout-feature correlations with the target
|Rank|Layout feature|Corr|Meaning|
|---|---|---|---|
|1|pack_station_count|-0.1862|Number of packing stations|
|2|robot_total|-0.1105|Total robot count|
|3|emergency_exit_count|-0.0443|Emergency exit count|
|4|zone_dispersion|-0.0267|How dispersed warehouse zones are|
|5|layout_compactness|-0.0220|Compactness of the layout|
|6|one_way_ratio|-0.0186|Share of one-way aisles|
|7|intersection_count|-0.0165|Number of intersections|
|8|building_age_years|-0.0137|Building age in years|
|9|charger_count|-0.0128|Number of chargers|
|10|fire_sprinkler_count|0.0098|Fire sprinkler count|

### Target by layout_type
|layout_type|count|mean delay|median delay|
|---|---|---|---|
|hub_spoke|43,375|22.2800|11.3177|
|hybrid|73,125|18.4119|8.2503|
|narrow|42,250|18.3608|9.4674|
|grid|91,250|18.1048|8.9241|

- Directly merge static layout variables such as `pack_station_count`, `robot_total`, `floor_area_sqm`, `charger_count`, and `layout_type`.
- Add normalized capacity features such as `robot_total / floor_area_sqm`, `charger_count / robot_total`, `pack_station_count / robot_total`, and `intersection_count / floor_area_sqm`.
- Because test contains unseen layouts, do not rely only on raw `layout_id` memorization. Layout metadata is important for generalization.

## scenario_id and Implicit Timeslot Structure
- No explicit `timeslot` column exists in train or test.
- Train: `10,000` scenarios x `25` rows each = `250,000` rows
- Test: `2,000` scenarios x `25` rows each = `50,000` rows
- Combined: `(10,000 + 2,000) x 25 = 300,000` rows, matching the full dataset exactly.
- `groupby("scenario_id").cumcount()` yields `0..24` in train and `0..24` in test.
- This supports sequence-aware feature engineering: implicit slot index, lag features, rolling means, differences, and per-scenario trend summaries.
- `sample_submission.csv` ID order matches `test.csv` exactly: `True`

## Recommended Feature Engineering and Modeling Directions
1. Create `implicit_timeslot` with `groupby("scenario_id").cumcount()` and add lag / rolling features inside each scenario.
2. Build demand-vs-capacity ratios such as `order_inflow_15m / robot_active`, `unique_sku_15m / pack_station_count`, and `order_wave_count / staff_on_floor`.
3. Add battery bottleneck interactions such as `low_battery_ratio * charge_queue_length`, `battery_mean - battery_std`, and `robot_charging / robot_active`.
4. Merge layout metadata and create normalized layout features such as robots per area, chargers per robot, and packing stations per robot.
5. Add missing-indicator features for high-missing columns and compare grouped imputation by `layout_id` or `scenario_id` against simple global imputations.
6. Use grouped validation by `scenario_id` to avoid leakage across the 25 rows of the same scenario.
7. Stress-test generalization to unseen layouts because test includes layouts not seen in train.
8. Good baseline models are CatBoost, LightGBM, or XGBoost regression, with optional `log1p(target)` and robust loss experiments.

## Extra Notes
- Removing rows with missing values is not viable because nearly every row contains at least one null.
- The combination of negative `robot_idle` correlation and positive `robot_charging` / `low_battery_ratio` correlation suggests that available robot capacity matters more than total nominal robot count.
- Among layout features, `pack_station_count` has the largest target correlation magnitude, which is consistent with packing capacity being a practical outbound bottleneck.