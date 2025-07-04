from process_bigraph.process_types import ProcessTypes

particle = {
    'type_name': 'string',
    'position': 'tuple[float,float,float]',
    'neighbor_ids': 'list[int]',
}

topology = {
    'type_name': 'string',
    'particle_ids': 'list[int]',
}

core = ProcessTypes()

# register types
core.register('topology', topology)
core.register('particle', particle)

core.register_process('readdy', ReaddyActinMembrane)
core.register_process('simularium-emitter', SimulariumEmitter)
