"""
Graph Optimization for Spintronic Neural Networks.

This module provides graph-level optimizations for neural networks
targeting spintronic hardware implementations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .pytorch_parser import SpintronicLayer, SpintronicModel


@dataclass
class OptimizationConfig:
    """Configuration for graph optimizations."""
    
    # Fusion options
    enable_conv_relu_fusion: bool = True
    enable_conv_bn_fusion: bool = True
    enable_linear_relu_fusion: bool = True
    
    # Quantization options
    enable_weight_quantization: bool = True
    enable_activation_quantization: bool = True
    weight_bits: int = 8
    activation_bits: int = 8
    
    # Hardware mapping options
    minimize_crossbars: bool = True
    balance_crossbar_utilization: bool = True
    enable_weight_sharing: bool = True
    
    # Energy optimization
    minimize_switching_energy: bool = True
    prefer_low_precision: bool = True


class GraphNode:
    """Represents a node in the computation graph."""
    
    def __init__(
        self,
        name: str,
        operation: str,
        layer: Optional[SpintronicLayer] = None
    ):
        self.name = name
        self.operation = operation
        self.layer = layer
        self.inputs: List['GraphNode'] = []
        self.outputs: List['GraphNode'] = []
        self.attributes: Dict[str, Any] = {}
        
    def add_input(self, node: 'GraphNode'):
        """Add input node."""
        if node not in self.inputs:
            self.inputs.append(node)
            node.outputs.append(self)
    
    def add_output(self, node: 'GraphNode'): 
        """Add output node."""
        if node not in self.outputs:
            self.outputs.append(node)
            node.inputs.append(self)
    
    def can_fuse_with(self, other: 'GraphNode') -> bool:
        """Check if this node can be fused with another."""
        fusable_pairs = [
            ('Conv2d', 'ReLU'),
            ('Linear', 'ReLU'),
            ('Conv2d', 'BatchNorm2d'),
        ]
        
        pair = (self.operation, other.operation)
        return pair in fusable_pairs


class ComputationGraph:
    """Represents the neural network computation graph."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.input_nodes: List[GraphNode] = []
        self.output_nodes: List[GraphNode] = []
    
    def add_node(self, node: GraphNode):
        """Add node to graph."""
        self.nodes[node.name] = node
    
    def get_topological_order(self) -> List[GraphNode]:
        """Get nodes in topological order."""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node: GraphNode):
            if node.name in temp_visited:
                raise ValueError("Graph has cycles")
            if node.name in visited:
                return
            
            temp_visited.add(node.name)
            
            for output_node in node.outputs:
                visit(output_node)
            
            temp_visited.remove(node.name)
            visited.add(node.name)
            result.append(node)
        
        # Visit all nodes
        for node in self.nodes.values():
            if node.name not in visited:
                visit(node)
        
        return result[::-1]  # Reverse for correct order
    
    def find_fusable_sequences(self) -> List[List[GraphNode]]:
        """Find sequences of nodes that can be fused."""
        sequences = []
        visited = set()
        
        for node in self.nodes.values():
            if node.name in visited:
                continue
                
            sequence = [node]
            current = node
            
            # Extend sequence forward
            while (len(current.outputs) == 1 and 
                   len(current.outputs[0].inputs) == 1 and
                   current.can_fuse_with(current.outputs[0])):
                next_node = current.outputs[0]
                sequence.append(next_node)
                current = next_node
            
            # Only add sequences with more than one node
            if len(sequence) > 1:
                sequences.append(sequence)
                for seq_node in sequence:
                    visited.add(seq_node.name)
        
        return sequences


class GraphOptimizer:
    """Optimizer for spintronic neural network computation graphs."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
    
    def optimize(self, model: SpintronicModel) -> SpintronicModel:
        """
        Apply graph-level optimizations to spintronic model.
        
        Args:
            model: Input spintronic model
            
        Returns:
            Optimized spintronic model
        """
        # Build computation graph
        graph = self._build_graph(model)
        
        # Apply optimizations
        if self.config.enable_conv_relu_fusion or self.config.enable_linear_relu_fusion:
            graph = self._apply_layer_fusion(graph)
        
        if self.config.enable_weight_quantization:
            graph = self._apply_weight_quantization(graph)
        
        if self.config.minimize_crossbars:
            graph = self._optimize_crossbar_mapping(graph)
        
        if self.config.minimize_switching_energy:
            graph = self._optimize_energy_efficiency(graph)
        
        # Convert back to SpintronicModel
        optimized_model = self._graph_to_model(graph, model)
        
        return optimized_model
    
    def _build_graph(self, model: SpintronicModel) -> ComputationGraph:
        """Build computation graph from spintronic model."""
        graph = ComputationGraph()
        prev_node = None
        
        for i, layer in enumerate(model.layers):
            node = GraphNode(
                name=f"node_{i}_{layer.name}",
                operation=layer.layer_type,
                layer=layer
            )
            
            graph.add_node(node)
            
            if prev_node is not None:
                prev_node.add_output(node)
            else:
                graph.input_nodes.append(node)
            
            prev_node = node
        
        if prev_node is not None:
            graph.output_nodes.append(prev_node)
        
        return graph
    
    def _apply_layer_fusion(self, graph: ComputationGraph) -> ComputationGraph:
        """Apply layer fusion optimizations."""
        fusable_sequences = graph.find_fusable_sequences()
        
        for sequence in fusable_sequences:
            if len(sequence) < 2:
                continue
            
            # Create fused layer
            fused_layer = self._fuse_layers(sequence)
            
            if fused_layer is not None:
                # Create new fused node
                fused_node = GraphNode(
                    name=f"fused_{sequence[0].name}_{sequence[-1].name}",
                    operation="Fused",
                    layer=fused_layer
                )
                
                # Connect inputs and outputs
                for input_node in sequence[0].inputs:
                    input_node.add_output(fused_node)
                
                for output_node in sequence[-1].outputs:
                    fused_node.add_output(output_node)
                
                # Remove old nodes
                for node in sequence:
                    if node.name in graph.nodes:
                        del graph.nodes[node.name]
                
                graph.add_node(fused_node)
        
        return graph
    
    def _fuse_layers(self, sequence: List[GraphNode]) -> Optional[SpintronicLayer]:
        """Fuse a sequence of layers into a single layer."""
        if len(sequence) < 2:
            return None
        
        first_layer = sequence[0].layer
        last_layer = sequence[-1].layer
        
        if first_layer is None:
            return None
        
        # Handle Conv2d + ReLU fusion
        if (sequence[0].operation == 'Conv2d' and 
            sequence[1].operation == 'ReLU'):
            
            fused_layer = SpintronicLayer(
                name=f"{first_layer.name}_relu_fused",
                layer_type="Conv2d_ReLU",
                input_shape=first_layer.input_shape,
                output_shape=last_layer.output_shape if last_layer else first_layer.output_shape,
                weights=first_layer.weights,
                bias=first_layer.bias
            )
            
            # Mark as having ReLU activation
            fused_layer.has_activation = True
            return fused_layer
        
        # Handle Linear + ReLU fusion
        elif (sequence[0].operation == 'Linear' and 
              sequence[1].operation == 'ReLU'):
            
            fused_layer = SpintronicLayer(
                name=f"{first_layer.name}_relu_fused",
                layer_type="Linear_ReLU", 
                input_shape=first_layer.input_shape,
                output_shape=last_layer.output_shape if last_layer else first_layer.output_shape,
                weights=first_layer.weights,
                bias=first_layer.bias
            )
            
            fused_layer.has_activation = True
            return fused_layer
        
        return None
    
    def _apply_weight_quantization(self, graph: ComputationGraph) -> ComputationGraph:
        """Apply weight quantization to reduce precision requirements."""
        for node in graph.nodes.values():
            if node.layer and node.layer.weights is not None:
                # Quantize weights
                quantized_weights = self._quantize_weights(
                    node.layer.weights,
                    self.config.weight_bits
                )
                node.layer.weights = quantized_weights
        
        return graph
    
    def _quantize_weights(self, weights: np.ndarray, bits: int) -> np.ndarray:
        """Quantize weight array to specified bit precision."""
        # Find weight range
        w_min, w_max = weights.min(), weights.max()
        
        # Symmetric quantization
        scale = max(abs(w_min), abs(w_max))
        
        # Quantization levels
        n_levels = 2 ** bits
        step_size = 2 * scale / (n_levels - 1)
        
        # Quantize
        quantized = np.round(weights / step_size) * step_size
        
        # Clip to range
        quantized = np.clip(quantized, -scale, scale)
        
        return quantized
    
    def _optimize_crossbar_mapping(self, graph: ComputationGraph) -> ComputationGraph:
        """Optimize how layers are mapped to crossbar arrays."""
        for node in graph.nodes.values():
            if node.layer and hasattr(node.layer, 'crossbars'):
                # Optimize crossbar utilization
                self._balance_crossbar_usage(node.layer)
        
        return graph
    
    def _balance_crossbar_usage(self, layer: SpintronicLayer):
        """Balance utilization across crossbar arrays."""
        if not layer.crossbars:
            return
        
        # Calculate current utilization
        utilizations = []
        for crossbar in layer.crossbars:
            total_cells = crossbar.rows * crossbar.cols
            # Utilization based on non-zero weights (simplified)
            if hasattr(crossbar, 'get_conductances'):
                conductances = crossbar.get_conductances()
                active_cells = np.count_nonzero(conductances)
                utilization = active_cells / total_cells
                utilizations.append(utilization)
        
        # If utilization is unbalanced, could redistribute weights
        # This is a simplified version - full implementation would
        # require more sophisticated weight redistribution
        avg_util = np.mean(utilizations) if utilizations else 0
        
        # Store utilization info for hardware generation
        layer.crossbar_utilization = avg_util
    
    def _optimize_energy_efficiency(self, graph: ComputationGraph) -> ComputationGraph:
        """Optimize for energy efficiency."""
        for node in graph.nodes.values():
            if node.layer and node.layer.weights is not None:
                # Minimize switching by preferring weights that map to
                # similar resistance states
                optimized_weights = self._optimize_weights_for_switching(
                    node.layer.weights
                )
                node.layer.weights = optimized_weights
        
        return graph
    
    def _optimize_weights_for_switching(self, weights: np.ndarray) -> np.ndarray:
        """Optimize weights to minimize switching energy."""
        # This is a simplified approach
        # In practice, would use more sophisticated algorithms
        
        # Cluster weights to reduce number of unique values
        from sklearn.cluster import KMeans
        
        flat_weights = weights.flatten()
        n_clusters = min(16, len(np.unique(flat_weights)))  # Limit unique values
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(flat_weights.reshape(-1, 1))
            
            # Replace weights with cluster centers
            optimized_flat = kmeans.cluster_centers_[clusters].flatten()
            optimized_weights = optimized_flat.reshape(weights.shape)
        else:
            optimized_weights = weights
        
        return optimized_weights
    
    def _graph_to_model(
        self, 
        graph: ComputationGraph, 
        original_model: SpintronicModel
    ) -> SpintronicModel:
        """Convert optimized graph back to SpintronicModel."""
        optimized_model = SpintronicModel(
            name=original_model.name + "_optimized",
            config=original_model.config
        )
        
        # Get nodes in topological order
        ordered_nodes = graph.get_topological_order()
        
        for node in ordered_nodes:
            if node.layer is not None:
                optimized_model.add_layer(node.layer)
        
        return optimized_model