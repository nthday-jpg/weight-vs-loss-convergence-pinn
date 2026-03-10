BALANCER_REGISTRY = {}

def register_balancer(name: str):
    """Decorator to register a balancer."""
    def decorator(cls):
        BALANCER_REGISTRY[name] = cls
        return cls
    return decorator

def create_balancer(name: str, **kwargs):
    """Factory function to create balancers."""
    if name not in BALANCER_REGISTRY:
        available = ', '.join(BALANCER_REGISTRY.keys())
        raise ValueError(f"Unknown balancer '{name}'. Available: {available}")
    return BALANCER_REGISTRY[name](**kwargs)

def list_balancers():
    """List all registered balancers."""
    return list(BALANCER_REGISTRY.keys())