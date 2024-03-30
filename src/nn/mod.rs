use crate::fundamental::Unit;

mod linear;
pub mod mlp;

pub trait Zeroable {
    fn zero_grad(&mut self) {
        for p in self.parameters() {
            p.zero_grad();
            //p.borrow_mut().grad = 0.0;
        }
    }

    fn parameters(&self) -> Vec<Unit>;
}
