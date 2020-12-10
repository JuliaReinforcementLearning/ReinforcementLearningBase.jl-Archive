# ReinforcementLearningBase.jl

[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl)

ReinforcementLearningBase.jl holds the common types and utility functions to be
shared by other components in ReinforcementLearning ecosystem.


## Examples

<table>
  <tr>
    <th></th>
    <th colspan="2">NumAgentStyle</th>
    <th colspan="2">DynamicStyle</th>
    <th colspan="2">ActionStyle</th>
    <th colspan="2">RewardStyle</th>
    <th colspan="2">InformationStyle</th>
    <th colspan="3">StateStyle</th>
    <th colspan="4">ChanceStyle</th>
    <th colspan="4">UtilityStyle</th>
  </tr>
  <tr>
    <th></th>
    <th>SINGLE_AGENT</th>
    <th>MultiAgent</th>
    <th>SEQUENTIAL </th>
    <th>SIMULTANEOUS</th>
    <th>MINIMAL_ACTION_SET </th>
    <th>FULL_ACTION_SET </th>
    <th>STEP_REWARD</th>
    <th>TERMINAL_REWARD</th>
    <th>PERFECT_INFORMATION</th>
    <th>IMPERFECT_INFORMATION</th>
    <th>Observation</th>
    <th>InternalState</th>
    <th>Information</th>
    <th>STOCHASTIC</th>
    <th>DETERMINISTIC</th>
    <th>EXPLICIT_STOCHASTIC</th>
    <th>SAMPLED_STOCHASTIC</th>
    <th>ZERO_SUM</th>
    <th>CONSTANT_SUM</th>
    <th>GENERAL_SUM</th>
    <th>IDENTICAL_REWARD</th>
  </tr>
  <tr>
  <tr>
    <td>MultiArmBanditsEnv</td>
    <td style="text-align:center">✔️<!-- SINGLE_AGENT --></td>
    <td style="text-align:center"> <!-- MultiAgent --> </td>
    <td style="text-align:center">✔️<!-- SEQUENTIAL  --> </td>
    <td style="text-align:center"> <!-- SIMULTANEOUS --> </td>
    <td style="text-align:center">✔️<!-- MINIMAL_ACTION_SET  --> </td>
    <td style="text-align:center"> <!-- FULL_ACTION_SET  --> </td>
    <td style="text-align:center"> <!-- STEP_REWARD --> </td>
    <td style="text-align:center">✔️<!-- TERMINAL_REWARD --> </td>
    <td style="text-align:center"> <!-- PERFECT_INFORMATION --> </td>
    <td style="text-align:center">✔️<!-- IMPERFECT_INFORMATION --> </td>
    <td style="text-align:center">✔️<!-- Observation --> </td>
    <td style="text-align:center"> <!-- InternalState --> </td>
    <td style="text-align:center"> <!-- Information --> </td>
    <td style="text-align:center">✔️<!-- STOCHASTIC --> </td>
    <td style="text-align:center"> <!-- DETERMINISTIC --> </td>
    <td style="text-align:center"> <!-- EXPLICIT_STOCHASTIC --> </td>
    <td style="text-align:center"> <!-- SAMPLED_STOCHASTIC --> </td>
    <td style="text-align:center"> <!-- ZERO_SUM --> </td>
    <td style="text-align:center"> <!-- CONSTANT_SUM --> </td>
    <td style="text-align:center">✔️<!-- GENERAL_SUM --> </td>
    <td style="text-align:center"> <!-- IDENTICAL_REWARD --> </td>
  </tr>
</table>