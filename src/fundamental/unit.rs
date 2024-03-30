use super::*;

use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct _Unit {
    _id: usize,
    pub data: f32,
    pub grad: f32,
    pub operation: Option<Operation>,
    pub children: Vec<Unit>,
}

impl Eq for _Unit {}
impl PartialEq for _Unit {
    fn eq(&self, other: &_Unit) -> bool {
        self._id == other._id
    }
}

impl Hash for _Unit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._id.hash(state);
    }
}

impl fmt::Debug for _Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("_Unit")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("operation", &self.operation)
            .field("children_size", &self.children.len())
            .finish()
    }
}

impl fmt::Display for _Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unit {{ data: {}, grad: {}, operation: {:?}, children_size: {} }}",
            self.data,
            self.grad,
            self.operation,
            self.children.len()
        )
    }
}

pub fn new_unit(data: f32) -> Unit {
    Unit::new(_Unit::from(data))
}

impl _Unit {
    pub fn new(data: f32, op: Option<Operation>, children: Vec<Unit>) -> _Unit {
        _Unit {
            _id: rand::random(),
            data,
            operation: op,
            children,
            grad: 0.0,
        }
    }

    pub fn from(data: f32) -> _Unit {
        _Unit {
            _id: rand::random(),
            data,
            grad: 0.0,
            operation: None,
            children: vec![],
        }
    }

    pub fn id(&self) -> usize {
        self._id
    }

    pub fn self_back_propagation(&mut self) {
        match self.operation {
            Some(ref op) => {
                match op {
                    Operation::Add(ref a, ref b) => {
                        a.borrow_mut().grad += self.grad;
                        b.borrow_mut().grad += self.grad;
                    }
                    Operation::Mul(ref a, ref b) => {
                        a.borrow_mut().grad += self.grad * b.borrow().data;
                        b.borrow_mut().grad += self.grad * a.borrow().data;
                    }
                    Operation::Tanh(ref x) => {
                        let tanh = x.borrow().data.tanh();
                        x.borrow_mut().grad += self.grad * (1.0 - tanh * tanh);
                    }
                    Operation::ReLU(ref x) => {
                        let relu = if self.data > 0.0 { 1.0 } else { 0.0 };
                        x.borrow_mut().grad += self.grad * relu;
                    }
                    Operation::Pow(ref a, b) => {
                        let mut inner_a = a.borrow_mut();
                        inner_a.grad += self.grad * b * inner_a.data.powf(b - 1.0);
                    }
                    Operation::Sub(ref a, ref b) => {
                        a.borrow_mut().grad += self.grad;
                        b.borrow_mut().grad -= self.grad;
                    }
                    _ => {}
                }
            }
            None => {
                // terminal unit
            }
        }
    }
}
