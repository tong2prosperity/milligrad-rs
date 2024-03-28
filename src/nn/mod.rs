use crate::fundamental::Unit;

mod linear;
pub mod mlp;

pub trait Zeroable {
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.borrow_mut().grad = 0.0;
        }
    }
    
    fn parameters(&self) -> Vec<Unit>;
}

