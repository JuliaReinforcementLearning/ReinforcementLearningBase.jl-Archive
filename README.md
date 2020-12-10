# ReinforcementLearningBase.jl

[![Build Status](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl.svg?branch=master)](https://travis-ci.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl)

ReinforcementLearningBase.jl holds the common types and utility functions to be
shared by other components in ReinforcementLearning ecosystem.


## Examples

<table>
  <tr align="center">
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
  <tr align="center">
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
  <tr align="center">
    <td>MultiArmBanditsEnv</td>
    <td>✔️<!-- SINGLE_AGENT --></td>
    <td> <!-- MultiAgent --> </td>
    <td>✔️<!-- SEQUENTIAL  --> </td>
    <td> <!-- SIMULTANEOUS --> </td>
    <td>✔️<!-- MINIMAL_ACTION_SET  --> </td>
    <td> <!-- FULL_ACTION_SET  --> </td>
    <td> <!-- STEP_REWARD --> </td>
    <td>✔️<!-- TERMINAL_REWARD --> </td>
    <td> <!-- PERFECT_INFORMATION --> </td>
    <td>✔️<!-- IMPERFECT_INFORMATION --> </td>
    <td>✔️<!-- Observation --> </td>
    <td> <!-- InternalState --> </td>
    <td> <!-- Information --> </td>
    <td>✔️<!-- STOCHASTIC --> </td>
    <td> <!-- DETERMINISTIC --> </td>
    <td> <!-- EXPLICIT_STOCHASTIC --> </td>
    <td> <!-- SAMPLED_STOCHASTIC --> </td>
    <td> <!-- ZERO_SUM --> </td>
    <td> <!-- CONSTANT_SUM --> </td>
    <td>✔️<!-- GENERAL_SUM --> </td>
    <td> <!-- IDENTICAL_REWARD --> </td>
  </tr>
</table>