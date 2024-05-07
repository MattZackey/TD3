# Twin Delayed Deep Deterministic Policy Gradient algorithm

The following is the [Twin Delayed Deep Deterministic Policy
Gradient algorithm (TD3)](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf) applied to the Walker2D and Half Cheetah environments from MuJoCo.

## Walker2D

<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Episode 100</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="Walker results/run100.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Episode 500</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Walker results/run500.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>

   <div style="flex: 1; text-align: center;">
    <h3>Episode 5000</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Walker results/run5000.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>


  \\
  The following figure shows the total reward the agent achieved for each espisode of training.

  ![Results](https://github.com/MattZackey/TD3/blob/main/Walker%20results/Walker%20results.png?raw=true)

## Half Cheetah
<div style="display: flex;">

  <div style="flex: 1; text-align: center;">
    <h3>Random policy</h3>
    <div style="border: 1px solid black; padding: 5px; display: inline-block">
      <img src="Cheetah results/Random Agent.gif" alt="Image 1" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  <div style="flex: 1; text-align: center;">
    <h3>Trained agent</h3>
    <div style="border: 1px solid black; padding: 5px;; display: inline-block">
      <img src="Cheetah results/Trained Agent.gif" alt="Image 2" style="max-width: 70%; width: 400px;">
    </div>
  </div>

  ![Results](https://github.com/MattZackey/TD3/blob/main/Cheetah%20results/Training%20results.png?raw=true)
