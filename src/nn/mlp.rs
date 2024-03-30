use std::iter::zip;
use crate::fundamental::Unit;
use super::*;
use rand::distributions::{Uniform, Distribution};
use crate::fundamental::op::*;
use crate::fundamental::unit::new_unit;

#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<Unit>,
    pub bias: Unit,
    pub activation: bool,
}

impl Neuron {
    pub fn new(input: u32, activation: bool) -> Self {
        let mut rng = rand::thread_rng();
        let between: Uniform<f32> = Uniform::from(-1.0..1.0);
        let mut weights = Vec::with_capacity(input as usize);
        for _ in 0..input {
            weights.push(new_unit(between.sample(&mut rng)));
        }

        Neuron {
            weights,
            bias: new_unit(0.0),
            activation,
        }
    }

    pub fn eval(&self, input: &Vec<Unit>) -> Unit {
        let mut sum = self.bias.clone();
        for (i, w) in zip(input.iter(), self.weights.iter()) {
            let n = mul(i, w);
            sum = add(&sum, &n);
        }
        if self.activation {
            sum = relu(&sum);
        }
        sum
    }
}

// impl FnMut<(Vec<Unit>, )> for Neuron {
//     extern "rust-call" fn call_mut(&mut self, args: (Vec<Unit>, )) -> Self::Output {
//         todo!()
//     }
// }
//
// impl FnOnce<(Vec<Unit>, )> for Neuron {
//     type Output = ();
//
//     extern "rust-call" fn call_once(self, args: (Vec<Unit>, )) -> Self::Output {
//         todo!()
//     }
// }


impl Zeroable for Neuron {
    fn parameters(&self) -> Vec<Unit> {
        let mut params = Vec::new();
        for w in self.weights.iter() {
            params.push(w.clone());
        }
        params.push(self.bias.clone());
        params
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input: u32, output: u32, activation: bool) -> Self {
        let mut neurons = Vec::with_capacity(output as usize);
        for _ in 0..output {
            neurons.push(Neuron::new(input, activation));
        }
        Layer {
            neurons,
        }
    }

    pub fn eval(&self, input: &Vec<Unit>) -> Vec<Unit> {
        let mut output = Vec::with_capacity(self.neurons.len());
        for n in self.neurons.iter() {
            output.push(n.eval(input));
        }
        output
    }
}

impl Zeroable for Layer {
    fn parameters(&self) -> Vec<Unit> {
        let mut params = Vec::new();
        for n in self.neurons.iter() {
            params.append(&mut n.parameters());
        }
        params
    }
}


pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(input: u32, hidden: Vec<u32>, output: u32) -> Self {
        let mut layers = Vec::with_capacity(hidden.len() + 1);
        let mut input = input;
        for &h in hidden.iter() {
            layers.push(Layer::new(input, h, true));
            input = h;
        }
        layers.push(Layer::new(input, output, false));
        MLP {
            layers,
        }
    }

    pub fn eval(&self, input: Vec<Unit>) -> Vec<Unit> {
        let mut output = input;
        for l in self.layers.iter() {
            output = l.eval(&output);
        }
        output
    }
}

impl Zeroable for MLP {
    fn parameters(&self) -> Vec<Unit> {
        let mut params = Vec::new();
        for l in self.layers.iter() {
            params.append(&mut l.parameters());
        }
        params
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron() {
        let n = Neuron::new(2, false);
        let input = vec![Unit::from(1.0), Unit::from(2.0)];
        let output = n.eval(&input);
        println!("neuron  is {:?}", n);
        println!("{:?}", output);
    }


    #[test]
    fn test_layer() {
        let l = Layer::new(4, 1, false);
        let input = vec![new_unit(1.0), new_unit(2.0), new_unit(3.0), new_unit(4.0)];
        let output = l.eval(&input);
        println!("layer is {:?}", l);
        println!("{:?}", output);
    }


    //接下来需要测试一下MLP在只有部分节点时候，权重更新是否正常
    #[test]
    fn test_mlp() {
        let mlp = MLP::new(2, vec![3, 4], 1);
        let input = vec![new_unit(1.0), new_unit(2.0)];
        let output = mlp.eval(input);
        let paran_num = mlp.parameters().len();

        println!("mlp output {:?}, mlp parameter size is {}", output, paran_num);
    }
    
    #[test]
    fn test_debug_simple_mlp() {
        let mlp = MLP::new(1, vec![2], 1);
        let parameters = mlp.parameters();
        for p in parameters.iter() {
            println!("{:?}", p.borrow());
        }
        
        let x = vec![new_unit(1.0), new_unit(2.0)];
        let y = vec![new_unit(2.0), new_unit(4.0)];
        
        let ypred = mlp.eval(x);
        //println!("ypred is {}",ypred);
    }

    #[test]
    fn test_mlp_ability() {
        let mut mlp = MLP::new(3, vec![4, 4], 1);
        let xs = vec![
            vec![new_unit(2.0), new_unit(3.0), new_unit(-1.0)],
            vec![new_unit(3.0), new_unit(-1.0), new_unit(0.5)],
            vec![new_unit(0.5), new_unit(1.0), new_unit(1.0)],
            vec![new_unit(1.0), new_unit(1.0), new_unit(-1.0)],
        ];

        let ys = vec![
            new_unit(1.0),
            new_unit(-1.0),
            new_unit(-1.0),
            new_unit(1.0),
        ];


        for k in 0..20 {
            let mut ypred = Vec::new();
            let xs_calc = xs.clone();
            for x in xs_calc {
                let o = mlp.eval(x).pop().unwrap();
                ypred.push(o);
            }
            //let ypred = xs.iter().flat_map(|x| mlp.eval(x)).collect::<Vec<Unit>>();
            // let loss_all = zip(ypred.iter(), ys.iter()).map(|(p, t)| {
            //     pow(&sub(p, t), 2.0)
            // }).collect::<Vec<_>>();

            let mut loss = Unit::from(0.0);
            for i in 0..ypred.len() {
                //let s = sub(&ypred[i], &ys[i]);
                let s= &ypred[i] - &ys[i];
                let p = pow(&s, 2.0);
                loss = &loss + &p;
            }
            let parameters = mlp.parameters();

            mlp.zero_grad();
            // for i in 1..4 {
            //     println!("{:?}", parameters[i].borrow());
            // }

            backward(&loss);
            // for i in 1..4 {
            //     println!("{:?}", parameters[i].borrow());
            // }

            println!("Before update iter {}, loss is {:?}", k, loss.borrow().data);

            // println!(" parameter length is {}", parameters.len());
            // for p in parameters.iter() {
            //     println!("{:?}", p.borrow());
            // }

            for p in parameters.iter() {
                let mut p = p.borrow_mut();
                p.data += -0.05 * p.grad;
            }

         //   println!("After update iter {}, loss is {:?}", k, loss);

            // for p in parameters {
            //     println!("{:?}", p.borrow());
            // }
        }
    }
}
