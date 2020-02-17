cdef class GameConfigCy:
    cdef public float bot_driving_speed, bot_radius, bot_turning_speed, noise_time, noise_angle, noise_distance, \
        noise_proximity, sensor_ray_distance, target_reached
    cdef public int batch, duration, max_game_id, max_eval_game_id, fps, p2m, x_axis, y_axis
    cdef public bint time_all
